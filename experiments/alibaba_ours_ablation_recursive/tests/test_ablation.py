from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent / "src"))
sys.path.insert(0, str(ROOT.parent / "alibaba_ours_lh_recursive" / "src"))
sys.path.insert(0, str(ROOT / "src"))

from alibaba_ours_exp.training import TrainingConfig
from alibaba_ours_ablation.model import (
    ABLATION_VARIANTS,
    AblationModel,
    AblationModelConfig,
    hypergraph_adjacency,
    pairwise_adjacency,
)
from alibaba_ours_ablation.direct import (
    _atomic_torch_save,
    _torch_load_path,
    compute_slot_membership_series,
    train_direct_model,
)
from alibaba_ours_ablation.runner import _suite_variants, materialize_variant_config
from alibaba_ours_ablation.suite import collect_suite, write_json_atomic


def config(variant: str) -> AblationModelConfig:
    return AblationModelConfig(
        variant=variant,
        resource_dim=4,
        num_states=4,
        history_window=5,
        prototype_length=5,
        max_order=3,
        message_passing_steps=2,
        message_hidden_dim=4,
        transition_hidden_dim=4,
        random_state=7,
    )


class AblationSemanticsTests(unittest.TestCase):
    def test_torch_artifacts_round_trip_under_unicode_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "工作负载" / "模型"
            checkpoint_path = root / "epoch-resume.pt"
            _atomic_torch_save(checkpoint_path, {"epoch": 1})
            checkpoint = _torch_load_path(checkpoint_path, map_location=torch.device("cpu"))
            self.assertEqual(checkpoint["epoch"], 1)

            model_path = root / "model.pt"
            AblationModel(config("ns_mem")).save(model_path, {"unicode": True})
            _, metadata = AblationModel.load(model_path)
            self.assertTrue(metadata["unicode"])

    def test_membership_is_defined_per_slot_and_independent_of_L(self) -> None:
        short = AblationModel(config("ns_mem"))
        long_config = config("ns_mem")
        long_config.history_window = 15
        long = AblationModel(long_config)
        long.shape_encoder.set_prototypes(short.shape_encoder.prototypes.detach())
        resources = torch.randn(3, 15, 4)
        mask = torch.ones(3, 15, dtype=torch.bool)
        short_value = short.encode_membership(resources[:, -5:], mask[:, -5:])
        long_value = long.encode_membership(resources, mask)
        torch.testing.assert_close(short_value, long_value)

    def test_direct_horizons_do_not_feed_predictions_back(self) -> None:
        model = AblationModel(config("ns_mem"))
        origin = torch.softmax(torch.randn(3, 4), dim=-1)
        contexts = torch.zeros(3, model.config.max_order, dtype=torch.long)
        together = model.direct_multi_step(
            origin,
            contexts,
            torch.tensor([[1.0, 2.0]]).expand(3, -1),
            torch.tensor([1, 2]),
        )["final"][:, 1]
        alone = model.direct_multi_step(
            origin,
            contexts,
            torch.tensor([[2.0]]).expand(3, -1),
            torch.tensor([2]),
        )["final"][:, 0]
        torch.testing.assert_close(together, alone)

    def test_vo_markov_direct_diagnostics_keep_common_report_width(self) -> None:
        model = AblationModel(config("vo_markov"))
        origin = torch.softmax(torch.randn(3, 4), dim=-1)
        output = model.direct_multi_step(
            origin,
            torch.zeros(3, 1, dtype=torch.long),
            torch.tensor([[1.0, 2.0]]).expand(3, -1),
            torch.tensor([1, 2]),
        )
        self.assertEqual(output["backoff_diagnostics"]["supports"].shape[-1], 4)

    def test_slot_membership_series_has_one_label_per_slot(self) -> None:
        model = AblationModel(config("ns_mem"))
        resources = torch.randn(3, 7, 4)
        memberships = compute_slot_membership_series(model, resources)
        self.assertEqual(tuple(memberships.shape), (3, 7, 4))
        torch.testing.assert_close(
            memberships.sum(dim=-1), torch.ones(3, 7), atol=1e-6, rtol=1e-6
        )

    def test_direct_training_covers_every_horizon_across_chunks(self) -> None:
        model_config = config("ns_mem")
        model_config.max_forecast_horizon = 12
        model = AblationModel(model_config)
        training = TrainingConfig(
            epochs=1,
            max_origins_per_epoch=1,
            early_stopping_patience=1,
            prototype_candidates=12,
            device="cpu",
            random_state=7,
        )
        training.direct_horizon_batch_size = 5
        result = train_direct_model(
            model,
            torch.rand(3, 24, 4),
            torch.arange(24, dtype=torch.float32),
            torch.ones(3, 24, dtype=torch.bool),
            config=training,
        )
        self.assertEqual(result.history[-1]["direct_horizons_trained"], 12.0)

    def test_structural_overrides_are_exact(self) -> None:
        self.assertEqual(config("vo_markov").max_order, 1)
        self.assertEqual(config("multi_hop").message_passing_steps, 1)
        for variant in set(ABLATION_VARIANTS) - {"vo_markov"}:
            self.assertEqual(config(variant).max_order, 3)
        for variant in set(ABLATION_VARIANTS) - {"multi_hop"}:
            self.assertEqual(config(variant).message_passing_steps, 2)

    def test_ns_mem_has_no_neighbour_membership_effect(self) -> None:
        model = AblationModel(config("ns_mem"))
        memberships = torch.softmax(torch.randn(3, 4), dim=-1)
        resources = torch.randn(3, 4)
        topology = torch.ones(3, 3) - torch.eye(3)
        spatial, diagnostics = model._spatial_distribution(
            memberships, resources, topology, None
        )
        torch.testing.assert_close(spatial, memberships)
        torch.testing.assert_close(diagnostics["modulation_l1"], torch.zeros(3))

    def test_dyn_trans_is_global_and_time_invariant(self) -> None:
        model = AblationModel(config("dyn_trans"))
        memberships = torch.softmax(torch.randn(3, 4), dim=-1)
        first, first_matrix = model._dynamic_distribution(
            memberships, torch.tensor([1.0, 2.0, 3.0])
        )
        second, second_matrix = model._dynamic_distribution(
            memberships, torch.tensor([100.0, 200.0, 300.0])
        )
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first_matrix, second_matrix)
        torch.testing.assert_close(first_matrix[0], first_matrix[2])

    def test_pairwise_and_hypergraph_weights_differ_for_large_hyperedge(self) -> None:
        deployment = np.asarray(
            [[1, 1, 1, 0], [0, 0, 1, 1]], dtype=np.float32
        )
        pairwise = pairwise_adjacency(deployment)
        hypergraph = hypergraph_adjacency(deployment)
        self.assertTrue(set(np.unique(pairwise)).issubset({0.0, 1.0}))
        self.assertFalse(np.array_equal(pairwise, hypergraph))
        np.testing.assert_array_equal(np.diag(hypergraph), np.zeros(4))


class SuiteConfigTests(unittest.TestCase):
    def test_ablation_json_writer_encodes_non_finite_metrics_as_null(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            write_json_atomic(
                path,
                {"nan": float("nan"), "inf": np.float64(float("inf")), "finite": 0.5},
            )
            value = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsNone(value["nan"])
        self.assertIsNone(value["inf"])
        self.assertEqual(value["finite"], 0.5)

    def test_third_machine_suite_is_complete(self) -> None:
        variants = _suite_variants(ROOT / "configs" / "ablation.json")
        self.assertEqual(sorted(variants), sorted(ABLATION_VARIANTS))
        self.assertEqual(len(variants), len(set(variants)))

    def test_collection_excludes_legacy_recursive_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            for name, strategy, membership in (
                ("legacy", "recursive", None),
                ("direct", "direct_multi_step", "per_slot"),
            ):
                run = runs / name
                run.mkdir(parents=True)
                write_json_atomic(
                    run / "run_metadata.json",
                    {
                        "ablation_variant": "ns_mem",
                        "stage": "aggregate_complete",
                        "formal_result": True,
                        "forecast_strategy": strategy,
                        "membership_definition": membership,
                    },
                )
            summary = collect_suite(runs, root / "reports")
        self.assertEqual(summary["runs_found"], 1)
        self.assertEqual(summary["formal_variants_complete"], ["ns_mem"])

    def test_generated_configs_only_add_ablation_identity(self) -> None:
        base_path = ROOT / "configs" / "ablation.json"
        base = json.loads(base_path.read_text(encoding="utf-8"))
        generated = materialize_variant_config(base_path, "vo_markov")
        value = json.loads(generated.read_text(encoding="utf-8"))
        self.assertEqual(value["ablation"]["variant"], "vo_markov")
        self.assertEqual(value["protocol"]["forecast_strategy"], "direct_multi_step")
        self.assertEqual(value["protocol"]["membership_definition"], "per_slot")
        self.assertEqual(value["protocol"]["target_window_steps"], 1)
        for key in ("time_axis", "split", "grid", "training", "origins", "sources"):
            self.assertEqual(value[key], base[key])


if __name__ == "__main__":
    unittest.main()
