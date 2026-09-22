from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alibaba_ours_duibi1.model import (
    DirectMembershipForecaster,
    DirectModelConfig,
    DirectPhysicalTimeBackoff,
    direct_ours_loss,
)


class DirectModelTests(unittest.TestCase):
    def _config(self, **updates: object) -> DirectModelConfig:
        values: dict[str, object] = {
            "input_dim": 8,
            "resource_dim": 3,
            "num_states": 4,
            "history_length": 5,
            "horizon_steps": 7,
            "workload_count": 3,
            "max_order": 3,
            "message_hidden_dim": 6,
            "transition_hidden_dim": 8,
            "transition_horizon_chunk": 3,
            "dropout": 0.0,
            "message_passing_steps": 2,
        }
        values.update(updates)
        return DirectModelConfig(**values)

    def test_direct_ours_shape_simplex_components_and_loss(self) -> None:
        config = self._config()
        deployment = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.float32)
        model = DirectMembershipForecaster(config, deployment)
        resources = torch.randn(2, 3, 5, 3)
        memberships = torch.softmax(torch.randn(2, 3, 5, 4), dim=-1)
        valid_feature = torch.ones(2, 3, 5, 1)
        history = torch.cat([resources, memberships, valid_feature], dim=-1)
        history_mask = torch.ones(2, 3, 5, dtype=torch.bool)
        origin_time = torch.tensor([1000.0, 1060.0])
        outputs = model(
            history,
            history_mask,
            origin_time,
            return_components=True,
        )
        prediction = outputs["final"]
        self.assertEqual(tuple(prediction.shape), (2, 3, 7, 4))
        self.assertTrue(
            torch.allclose(prediction.sum(dim=-1), torch.ones(2, 3, 7), atol=1e-5)
        )
        self.assertEqual(
            tuple(outputs["backoff_diagnostics"]["order_weights"].shape),
            (2, 3, 7, 4),
        )
        target = torch.softmax(torch.randn_like(prediction), dim=-1)
        valid = torch.ones(2, 3, 7, dtype=torch.bool)
        loss, parts = direct_ours_loss(outputs, target, valid)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts["valid_targets"], 42)

    def test_backoff_uses_fixed_real_history_for_every_horizon(self) -> None:
        config = self._config(
            input_dim=5,
            resource_dim=2,
            num_states=2,
            workload_count=1,
            horizon_steps=5,
            max_order=1,
            backoff_decay=0.0,
            backoff_smoothing=1.0,
        )
        backoff = DirectPhysicalTimeBackoff(config)
        hard = torch.tensor([0, 1, 0, 1, 0])
        memberships = torch.nn.functional.one_hot(hard, num_classes=2).float()
        memberships = memberships.view(1, 1, 5, 2)
        result = backoff(memberships, torch.ones(1, 1, 5, dtype=torch.bool))
        order_one = result["order_distributions"][0, 0, :, 1]
        self.assertTrue(torch.allclose(order_one[0], torch.tensor([0.25, 0.75])))
        self.assertTrue(
            torch.allclose(order_one[2], torch.tensor([1.0 / 3.0, 2.0 / 3.0]))
        )
        self.assertTrue(torch.allclose(order_one[4], torch.tensor([0.5, 0.5])))
        self.assertEqual(float(result["supports"][0, 0, 0, 1]), 2.0)
        self.assertEqual(float(result["supports"][0, 0, 2, 1]), 1.0)
        self.assertEqual(float(result["supports"][0, 0, 4, 1]), 0.0)

    def test_lambda_is_applied_in_physical_slot_units(self) -> None:
        config = self._config(
            input_dim=5,
            resource_dim=2,
            num_states=2,
            workload_count=1,
            horizon_steps=2,
            max_order=1,
            backoff_decay=0.02,
        )
        backoff = DirectPhysicalTimeBackoff(config)
        hard = torch.tensor([0, 1, 0, 1, 0])
        memberships = torch.nn.functional.one_hot(hard, num_classes=2).float()
        result = backoff(
            memberships.view(1, 1, 5, 2),
            torch.ones(1, 1, 5, dtype=torch.bool),
        )
        supports = result["supports"][0, 0, :, 1]
        self.assertGreater(float(supports[0]), float(supports[1]))

    def test_backoff_context_skips_missing_slots_like_original_ours(self) -> None:
        config = self._config(
            input_dim=5,
            resource_dim=2,
            num_states=2,
            workload_count=1,
            horizon_steps=1,
            max_order=1,
            backoff_decay=0.0,
        )
        backoff = DirectPhysicalTimeBackoff(config)
        hard = torch.tensor([0, 0, 1, 0, 0])
        memberships = torch.nn.functional.one_hot(hard, num_classes=2).float()
        valid = torch.tensor([True, False, True, False, True]).view(1, 1, 5)
        result = backoff(memberships.view(1, 1, 5, 2), valid)
        # The valid sequence is [0, 1, 0]. Query context 0 therefore matches
        # the transition 0->1 at physical position 2 despite the missing slot.
        self.assertEqual(float(result["supports"][0, 0, 0, 1]), 1.0)
        self.assertTrue(
            torch.allclose(
                result["order_distributions"][0, 0, 0, 1],
                torch.tensor([1.0 / 3.0, 2.0 / 3.0]),
            )
        )


if __name__ == "__main__":
    unittest.main()
