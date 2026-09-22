from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import numpy as np

from .config import ExperimentConfig, TaskSpec
from .data import PreparedData, prepare_data, valid_origins
from .metrics import write_metric_bundle
from .model import (
    DirectMembershipForecaster,
    DirectModelConfig,
    direct_ours_loss,
)
from .progress import ProgressBar, print_stage
from .utils import (
    atomic_save_npz,
    atomic_write_csv,
    atomic_write_json,
    code_hash,
    cpu_state_dict,
    optimizer_to,
    resolve_device,
    seed_everything,
    stable_seed,
)


class ExperimentError(RuntimeError):
    pass


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024**2))
    except Exception:
        return None


def _subset_data(data: PreparedData, count: int) -> PreparedData:
    selected = slice(0, min(int(count), data.workloads))
    deployment = data.deployment[:, selected]
    active_nodes = deployment.sum(axis=1) > 0
    if active_nodes.any():
        deployment = deployment[active_nodes]
    else:
        deployment = np.zeros((1, min(int(count), data.workloads)), dtype=np.float32)
    return PreparedData(
        workload_ids=data.workload_ids[selected],
        resource_names=data.resource_names,
        deployment=deployment,
        input_normalized=data.input_normalized[selected],
        input_memberships=data.input_memberships[selected],
        target_memberships=data.target_memberships[selected],
        observed_values=data.observed_values[selected],
        input_mask=data.input_mask[selected],
        target_mask=data.target_mask[selected],
        time_seconds=data.time_seconds,
        resource_mean=data.resource_mean,
        resource_std=data.resource_std,
        centroids_normalized=data.centroids_normalized,
        membership_scale2=data.membership_scale2,
        train_end=data.train_end,
        validation_end=data.validation_end,
        cache_path=data.cache_path,
        cache_signature=data.cache_signature,
    )


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        run_id: str,
        resume: bool,
        smoke: bool = False,
    ) -> None:
        if not run_id or any(value in run_id for value in ("/", "\\", "..")):
            raise ExperimentError("run-id 不能为空且不能包含路径分隔符")
        self.config = config
        self.run_id = run_id
        self.resume = bool(resume)
        self.smoke = bool(smoke)
        self.run_dir = config.output_dir / run_id
        self.manifest_path = self.run_dir / "manifest.json"
        self.state_path = self.run_dir / "state.json"
        self.device = resolve_device(str(config.training.get("device", "cpu")))
        self.source_hash = code_hash(config.root)
        self.interval = float(config.runtime.get("progress_interval_seconds", 2.0))

    def task_queue(self) -> list[TaskSpec]:
        if not self.smoke:
            return self.config.task_queue()
        horizon_by_name = {item.name: item for item in self.config.horizons}
        return [
            TaskSpec(length, horizon_by_name[name])
            for length in (1440, 5)
            for name in ("long_1p2d", "medium_8h", "short_60m")
        ]

    def _signature(self) -> str:
        suffix = (
            ":smoke-v4-direct-ours-backoff"
            if self.smoke
            else ":formal-v3-direct-ours-backoff"
        )
        return self.config.experiment_signature() + suffix

    def _initial_state(self, tasks: Sequence[TaskSpec]) -> dict[str, Any]:
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return {
            "run_id": self.run_id,
            "mode": "smoke" if self.smoke else "formal",
            "created_at": now,
            "updated_at": now,
            "tasks": [
                {
                    **task.as_dict(),
                    "queue_index": index + 1,
                    "status": "pending",
                    "last_error": None,
                }
                for index, task in enumerate(tasks)
            ],
        }

    def ensure_run(self, tasks: Sequence[TaskSpec]) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        signature = self._signature()
        if self.manifest_path.is_file():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("experiment_signature") != signature:
                raise ExperimentError(
                    "现有run-id对应的配置签名不同。为避免混用结果，请更换run-id。"
                )
            if manifest.get("source_hash") != self.source_hash:
                print_stage("警告：源码哈希已变化；run-id保持不变，兼容检查通过后继续断点。")
        else:
            if any(self.run_dir.iterdir()) and not self.resume:
                raise ExperimentError("运行目录非空；请使用--resume或更换run-id")
            atomic_write_json(
                self.manifest_path,
                {
                    "run_id": self.run_id,
                    "mode": "smoke" if self.smoke else "formal",
                    "experiment_signature": signature,
                    "source_hash": self.source_hash,
                    "config_path": self.config.path,
                    "dataset_path": self.config.dataset_path,
                    "forecast_strategy": "direct_multi_step",
                    "model_family": "Ours-TopologyOnly-Direct",
                    "recursive_feedback": False,
                    "time_step_seconds": 60,
                    "task_queue": [task.as_dict() for task in tasks],
                    "source_hash_is_audit_only": True,
                },
            )
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            existing = [item["task_id"] for item in state.get("tasks", [])]
            expected = [item.task_id for item in tasks]
            if existing != expected:
                raise ExperimentError("state.json中的任务顺序与当前配置不一致")
            return state
        state = self._initial_state(tasks)
        atomic_write_json(self.state_path, state)
        return state

    def _load_state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _update_task_state(self, task: TaskSpec, **updates: Any) -> None:
        state = self._load_state()
        found = False
        for item in state["tasks"]:
            if item["task_id"] == task.task_id:
                item.update(updates)
                found = True
                break
        if not found:
            raise ExperimentError(f"state.json中找不到任务{task.task_id}")
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_write_json(self.state_path, state)

    def _task_dir(self, task: TaskSpec) -> Path:
        return self.run_dir / "tasks" / task.task_id

    def _model_config(self, task: TaskSpec, data: PreparedData) -> DirectModelConfig:
        model = self.config.model
        # Fixed per-slot fuzzy memberships are shared across every L/H model.
        # The model also receives normalized resources and the causal valid flag.
        input_dim = data.resources + data.num_states + 1
        return DirectModelConfig(
            input_dim=input_dim,
            resource_dim=data.resources,
            num_states=data.num_states,
            history_length=task.history_length,
            horizon_steps=task.horizon.steps,
            workload_count=data.workloads,
            max_order=int(model.get("max_order", 3)),
            backoff_decay=float(model.get("time_decay", 0.02)),
            backoff_time_unit_steps=float(model.get("backoff_time_unit_steps", 1.0)),
            backoff_smoothing=float(model.get("backoff_smoothing", 1.0)),
            backoff_threshold=float(model.get("backoff_threshold", 2.0)),
            backoff_temperature=float(model.get("backoff_temperature", 1.0)),
            backoff_blend=float(model.get("backoff_blend", 0.5)),
            time_period_steps=float(model.get("time_period_steps", 1440.0)),
            einstein_strength=float(model.get("einstein_strength", 0.5)),
            message_passing_steps=int(model.get("message_passing_steps", 2)),
            message_hidden_dim=int(model.get("message_hidden_dim", 8)),
            transition_hidden_dim=int(model.get("transition_hidden_dim", 32)),
            transition_horizon_chunk=int(model.get("transition_horizon_chunk", 240)),
            dropout=float(model.get("dropout", 0.0)),
            time_step_seconds=self.config.time_step_seconds,
        )

    def _batch(
        self,
        data: PreparedData,
        task: TaskSpec,
        origins: Sequence[int],
        *,
        include_targets: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        histories: list[np.ndarray] = []
        history_masks: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        target_masks: list[np.ndarray] = []
        origin_times: list[float] = []
        for origin in origins:
            start = int(origin) - task.history_length + 1
            if start < 0:
                raise ExperimentError("历史窗口越过数据起点")
            resource = data.input_normalized[:, start : origin + 1]
            membership = data.input_memberships[:, start : origin + 1]
            valid = data.input_mask[:, start : origin + 1]
            feature = np.concatenate(
                [
                    resource,
                    membership,
                    valid[:, :, None].astype(np.float32),
                ],
                axis=-1,
            )
            histories.append(np.ascontiguousarray(feature, dtype=np.float32))
            history_masks.append(np.ascontiguousarray(valid, dtype=bool))
            origin_times.append(float(data.time_seconds[origin]))
            if include_targets:
                stop = origin + task.horizon.steps + 1
                targets.append(data.target_memberships[:, origin + 1 : stop])
                target_masks.append(data.target_mask[:, origin + 1 : stop])
        history_tensor = torch.from_numpy(np.stack(histories)).to(self.device)
        history_mask_tensor = torch.from_numpy(np.stack(history_masks)).to(self.device)
        origin_time_tensor = torch.as_tensor(origin_times, dtype=torch.float32, device=self.device)
        if include_targets:
            target_tensor = torch.from_numpy(np.stack(targets).astype(np.float32)).to(self.device)
            target_mask_tensor = torch.from_numpy(np.stack(target_masks)).to(self.device)
        else:
            target_tensor = None
            target_mask_tensor = None
        return (
            history_tensor,
            history_mask_tensor,
            origin_time_tensor,
            target_tensor,
            target_mask_tensor,
        )

    def _available_origins(self, data: PreparedData, task: TaskSpec, split: str) -> list[int]:
        stride = int(self.config.evaluation.get("origin_stride_steps", 60))
        start, stop = data.split_bounds(split)
        origin_history_length = (
            max(self.config.history_lengths) if split == "train" else task.history_length
        )
        return valid_origins(
            split_start=start,
            split_end=stop,
            history_length=origin_history_length,
            horizon_steps=task.horizon.steps,
            stride=stride,
            training=(split == "train"),
        )

    def _common_origins(self, data: PreparedData, split: str) -> list[int]:
        longest = max(self.config.horizons, key=lambda value: value.steps)
        start, stop = data.split_bounds(split)
        return valid_origins(
            split_start=start,
            split_end=stop,
            history_length=max(self.config.history_lengths),
            horizon_steps=longest.steps,
            stride=int(self.config.evaluation.get("origin_stride_steps", 60)),
            training=False,
        )

    def _training_settings(self, task: TaskSpec) -> dict[str, Any]:
        settings = dict(self.config.training)
        if self.smoke:
            settings.update(
                {
                    "epochs": 1,
                    "max_origins_per_epoch": 1,
                    "validation_origins": 1,
                    "early_stopping_patience": 1,
                }
            )
            settings["batch_origins"] = {task.horizon.name: 1}
        return settings

    def _selected_epoch_origins(
        self,
        available: Sequence[int],
        *,
        maximum: int,
        task: TaskSpec,
        epoch: int,
    ) -> list[int]:
        if len(available) <= maximum:
            selected = list(available)
        else:
            rng = np.random.default_rng(
                stable_seed(self.config.seed, f"{task.horizon.name}:epoch:{epoch}")
            )
            selected = [int(value) for value in rng.choice(available, maximum, replace=False)]
        rng = np.random.default_rng(
            stable_seed(self.config.seed, f"{task.horizon.name}:shuffle:{epoch}")
        )
        rng.shuffle(selected)
        return selected

    def _validation_loss(
        self,
        model: DirectMembershipForecaster,
        data: PreparedData,
        task: TaskSpec,
        origins: Sequence[int],
        brier_weight: float,
    ) -> float:
        model.eval()
        losses: list[float] = []
        with torch.inference_mode():
            for origin in origins:
                history, history_mask, origin_time, target, target_mask = self._batch(
                    data, task, [origin], include_targets=True
                )
                outputs = model(
                    history,
                    history_mask,
                    origin_time,
                    return_components=True,
                )
                loss, _ = direct_ours_loss(
                    outputs,
                    target,
                    target_mask,
                    dynamic_weight=float(
                        self.config.training.get("dynamic_loss_weight", 0.25)
                    ),
                    adaptive_order_weight=float(
                        self.config.training.get("adaptive_order_loss_weight", 0.25)
                    ),
                    brier_weight=brier_weight,
                    blend=float(model.config.backoff_blend),
                )
                losses.append(float(loss.detach().cpu()))
        if not losses:
            raise ExperimentError(f"{task.task_id}没有可用验证起点")
        return float(np.mean(losses))

    def _save_checkpoint(
        self,
        path: Path,
        *,
        model: DirectMembershipForecaster,
        optimizer: torch.optim.Optimizer,
        task: TaskSpec,
        epoch: int,
        best_validation_loss: float,
        patience_count: int,
        early_stopped: bool,
    ) -> None:
        checkpoint = {
            "task": task.as_dict(),
            "model_config": model.config.as_dict(),
            "cache_signature": getattr(self, "_cache_signature", None),
            "epoch": int(epoch),
            "best_validation_loss": float(best_validation_loss),
            "patience_count": int(patience_count),
            "early_stopped": bool(early_stopped),
            "model_state": cpu_state_dict(model),
            "optimizer_state": _cpu_tree(optimizer.state_dict()),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        _atomic_torch_save(path, checkpoint)

    def _restore_rng(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        if checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])

    def train_task(
        self, data: PreparedData, task: TaskSpec
    ) -> tuple[DirectMembershipForecaster, dict[str, Any]]:
        task_dir = self._task_dir(task)
        model_path = task_dir / "models" / "model.pt"
        settings = self._training_settings(task)
        model_config = self._model_config(task, data)
        # Reset every independent model to the same declared experimental seed.
        # This prevents queue order and arbitrary task names from becoming an
        # extra source of variation in the L/H comparison.
        task_seed = self.config.seed
        seed_everything(task_seed)
        model = DirectMembershipForecaster(
            model_config,
            torch.from_numpy(data.deployment),
        )
        # Fit the fixed transition prior on the chronological training split
        # only. It is stored in every checkpoint with the trainable parameters.
        model.set_transition_prior_from_memberships(
            torch.from_numpy(data.input_memberships[:, : data.train_end]),
            torch.from_numpy(data.input_mask[:, : data.train_end]),
        )
        model.to(self.device)
        if model_path.is_file():
            saved = _torch_load(model_path, map_location="cpu")
            if saved.get("model_config") != model_config.as_dict():
                raise ExperimentError(f"{task.task_id}已保存模型的结构与当前配置不一致")
            model.load_state_dict(saved["model_state"])
            model.to(self.device)
            return model, dict(saved.get("metadata", {}))

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(settings.get("learning_rate", 1e-3)),
            weight_decay=float(settings.get("weight_decay", 1e-4)),
        )
        checkpoint_dir = task_dir / "checkpoints"
        last_path = checkpoint_dir / "last.pt"
        best_path = checkpoint_dir / "best.pt"
        start_epoch = 0
        best_loss = math.inf
        patience_count = 0
        early_stopped = False
        history_path = task_dir / "training_history.csv"
        history_rows: list[dict[str, Any]] = []
        if history_path.is_file():
            with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
                history_rows = list(csv.DictReader(handle))
        if last_path.is_file() and self.resume:
            checkpoint = _torch_load(last_path, map_location="cpu")
            if checkpoint.get("model_config") != model_config.as_dict():
                raise ExperimentError(f"{task.task_id}训练断点的模型结构不兼容")
            if checkpoint.get("cache_signature") != data.cache_signature:
                raise ExperimentError(f"{task.task_id}训练断点的数据缓存不兼容")
            model.load_state_dict(checkpoint["model_state"])
            model.to(self.device)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer_to(optimizer, self.device)
            start_epoch = int(checkpoint["epoch"])
            best_loss = float(checkpoint["best_validation_loss"])
            patience_count = int(checkpoint["patience_count"])
            early_stopped = bool(checkpoint.get("early_stopped", False))
            self._restore_rng(checkpoint)
            print_stage(f"[{task.task_id}] 从epoch={start_epoch}恢复训练")

        train_origins = self._available_origins(data, task, "train")
        validation_origins = self._available_origins(data, task, "validation")
        if not train_origins or not validation_origins:
            raise ExperimentError(f"{task.task_id}没有足够的训练或验证起点")
        validation_count = min(int(settings.get("validation_origins", 4)), len(validation_origins))
        validation_selected = [
            validation_origins[index]
            for index in np.linspace(0, len(validation_origins) - 1, validation_count, dtype=int)
        ]
        epochs = int(settings.get("epochs", 20))
        maximum_origins = min(
            int(settings.get("max_origins_per_epoch", 64)), len(train_origins)
        )
        batch_size = int(settings.get("batch_origins", {}).get(task.horizon.name, 1))
        batches_per_epoch = math.ceil(maximum_origins / batch_size)
        progress = ProgressBar(
            f"train {task.task_id}",
            max((epochs - start_epoch) * batches_per_epoch, 1),
            interval_seconds=self.interval,
        )
        progress_units = 0
        brier_weight = float(settings.get("brier_loss_weight", 0.0))
        dynamic_weight = float(settings.get("dynamic_loss_weight", 0.25))
        adaptive_order_weight = float(
            settings.get("adaptive_order_loss_weight", 0.25)
        )
        minimum_delta = float(settings.get("minimum_delta", 1e-5))
        patience_limit = int(settings.get("early_stopping_patience", 4))
        for epoch in range(start_epoch, epochs):
            if early_stopped:
                break
            model.train()
            selected = self._selected_epoch_origins(
                train_origins,
                maximum=maximum_origins,
                task=task,
                epoch=epoch,
            )
            epoch_losses: list[float] = []
            epoch_started = time.perf_counter()
            for offset in range(0, len(selected), batch_size):
                batch_origins = selected[offset : offset + batch_size]
                history, history_mask, origin_time, target, target_mask = self._batch(
                    data, task, batch_origins, include_targets=True
                )
                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    history,
                    history_mask,
                    origin_time,
                    return_components=True,
                )
                prediction = outputs["final"]
                if not bool(torch.isfinite(prediction).all()):
                    raise ExperimentError(f"{task.task_id}产生NaN/Inf预测")
                loss, _ = direct_ours_loss(
                    outputs,
                    target,
                    target_mask,
                    dynamic_weight=dynamic_weight,
                    adaptive_order_weight=adaptive_order_weight,
                    brier_weight=brier_weight,
                    blend=float(model.config.backoff_blend),
                )
                if not bool(torch.isfinite(loss)):
                    raise ExperimentError(f"{task.task_id}训练损失为NaN/Inf")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(settings.get("gradient_clip", 5.0))
                )
                if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
                    raise ExperimentError(f"{task.task_id}梯度为NaN/Inf")
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
                progress_units += 1
                progress.update(
                    progress_units,
                    suffix=f"epoch={epoch + 1}/{epochs} batch={offset // batch_size + 1}/{batches_per_epoch}",
                )
            validation_loss = self._validation_loss(
                model, data, task, validation_selected, brier_weight
            )
            train_loss = float(np.mean(epoch_losses))
            improved = validation_loss < best_loss - minimum_delta
            if improved:
                best_loss = validation_loss
                patience_count = 0
                self._save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    task=task,
                    epoch=epoch + 1,
                    best_validation_loss=best_loss,
                    patience_count=patience_count,
                    early_stopped=False,
                )
            else:
                patience_count += 1
            early_stopped = patience_count >= patience_limit
            history_rows.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "best_validation_loss": best_loss,
                    "improved": improved,
                    "patience_count": patience_count,
                    "origins": len(selected),
                    "elapsed_seconds": time.perf_counter() - epoch_started,
                    "device": str(self.device),
                }
            )
            atomic_write_csv(history_path, history_rows)
            self._save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                task=task,
                epoch=epoch + 1,
                best_validation_loss=best_loss,
                patience_count=patience_count,
                early_stopped=early_stopped,
            )
            print_stage(
                f"[{task.task_id}] epoch={epoch + 1} train={train_loss:.6f} "
                f"validation={validation_loss:.6f} best={best_loss:.6f}"
            )
        progress.update(progress.total, suffix="early-stop" if early_stopped else "done", force=True)
        if not best_path.is_file():
            raise ExperimentError(f"{task.task_id}没有生成best.pt")
        best = _torch_load(best_path, map_location="cpu")
        model.load_state_dict(best["model_state"])
        model.to(self.device)
        metadata = {
            "task": task.as_dict(),
            "best_epoch": int(best["epoch"]),
            "best_validation_loss": float(best["best_validation_loss"]),
            "early_stopped": early_stopped,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "device": str(self.device),
            "seed": task_seed,
            "model_family": "Ours-TopologyOnly-Direct",
            "forecast_strategy": "direct_multi_step",
            "recursive_feedback": False,
            "max_order": model.config.max_order,
            "time_decay": model.config.backoff_decay,
            "backoff_blend": model.config.backoff_blend,
            "message_passing_steps": model.config.message_passing_steps,
        }
        _atomic_torch_save(
            model_path,
            {
                "model_config": model_config.as_dict(),
                "model_state": cpu_state_dict(model),
                "metadata": metadata,
                "cache_signature": data.cache_signature,
            },
        )
        atomic_write_json(task_dir / "models" / "metadata.json", metadata)
        return model, metadata

    def _prediction_path(self, task: TaskSpec, split: str, origin: int) -> Path:
        return self._task_dir(task) / "predictions" / split / f"origin-{origin:06d}.npz"

    def _prediction_is_valid(self, path: Path, task: TaskSpec, data: PreparedData) -> bool:
        if not path.is_file():
            return False
        try:
            with np.load(path, allow_pickle=False) as archive:
                return (
                    archive["predicted_membership"].shape
                    == (data.workloads, task.horizon.steps, data.num_states)
                    and archive["true_membership"].shape
                    == (data.workloads, task.horizon.steps, data.num_states)
                    and archive["valid_mask"].shape
                    == (data.workloads, task.horizon.steps)
                    and np.isfinite(archive["predicted_membership"]).all()
                )
        except Exception:
            return False

    def evaluate_split(
        self,
        model: DirectMembershipForecaster,
        data: PreparedData,
        task: TaskSpec,
        split: str,
    ) -> tuple[list[int], dict[str, Any]]:
        origins = self._available_origins(data, task, split)
        if self.smoke:
            origins = origins[:1]
        if not origins:
            raise ExperimentError(f"{task.task_id}在{split}没有评估起点")
        existing = sum(
            self._prediction_is_valid(self._prediction_path(task, split, origin), task, data)
            for origin in origins
        )
        progress = ProgressBar(
            f"evaluate {task.task_id} {split}",
            len(origins),
            interval_seconds=self.interval,
        )
        if existing:
            progress.update(existing, suffix="resume-scan", force=True)
        model.eval()
        records: list[dict[str, Any]] = []
        completed = existing
        for origin in origins:
            destination = self._prediction_path(task, split, origin)
            if self._prediction_is_valid(destination, task, data):
                with np.load(destination, allow_pickle=False) as archive:
                    records.append(
                        {
                            "origin": origin,
                            "latency_seconds": float(archive["latency_seconds"].item()),
                            "rss_mb": float(archive["rss_mb"].item()),
                            "gpu_peak_mb": float(archive["gpu_peak_mb"].item()),
                            "resumed": True,
                        }
                    )
                continue
            if destination.exists():
                destination.unlink()
            history, history_mask, origin_time, _, _ = self._batch(
                data, task, [origin], include_targets=False
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = model(
                    history,
                    history_mask,
                    origin_time,
                    return_components=True,
                )
                prediction = outputs["final"]
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                gpu_peak_mb = float(torch.cuda.max_memory_allocated(self.device) / (1024**2))
            else:
                gpu_peak_mb = 0.0
            latency = time.perf_counter() - started
            prediction_numpy = prediction[0].detach().cpu().numpy()
            backoff_diagnostics = outputs["backoff_diagnostics"]
            backoff_support_mean = (
                backoff_diagnostics["supports"][0].mean(dim=0).detach().cpu().numpy()
            )
            backoff_order_weight_mean = (
                backoff_diagnostics["order_weights"][0]
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
            )
            backoff_effective_order_mean = (
                backoff_diagnostics["effective_order"][0]
                .to(torch.float32)
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
            )
            dynamic_backoff_l1_mean = (
                (outputs["dynamic"][0] - outputs["backoff"][0])
                .abs()
                .sum(dim=-1)
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
            )
            graph_modulation_l1 = (
                outputs["graph_diagnostics"]["modulation_l1"][0]
                .detach()
                .cpu()
                .numpy()
            )
            if not np.isfinite(prediction_numpy).all():
                raise ExperimentError(f"{task.task_id}评估产生NaN/Inf")
            stop = origin + task.horizon.steps + 1
            true_membership = data.target_memberships[:, origin + 1 : stop]
            valid_mask = data.target_mask[:, origin + 1 : stop]
            dtype_name = str(self.config.evaluation.get("save_prediction_dtype", "float32"))
            save_dtype = np.float16 if dtype_name == "float16" else np.float32
            rss = _rss_mb()
            atomic_save_npz(
                destination,
                compressed=True,
                origin=np.asarray(origin, dtype=np.int64),
                origin_time_seconds=np.asarray(data.time_seconds[origin], dtype=np.int64),
                predicted_membership=prediction_numpy.astype(save_dtype),
                true_membership=true_membership.astype(save_dtype),
                valid_mask=valid_mask.astype(np.uint8),
                backoff_support_mean=backoff_support_mean.astype(np.float32),
                backoff_order_weight_mean=backoff_order_weight_mean.astype(np.float32),
                backoff_effective_order_mean=backoff_effective_order_mean.astype(np.float32),
                dynamic_backoff_l1_mean=dynamic_backoff_l1_mean.astype(np.float32),
                graph_modulation_l1=graph_modulation_l1.astype(np.float32),
                latency_seconds=np.asarray(latency, dtype=np.float64),
                rss_mb=np.asarray(rss if rss is not None else np.nan, dtype=np.float64),
                gpu_peak_mb=np.asarray(gpu_peak_mb, dtype=np.float64),
            )
            records.append(
                {
                    "origin": origin,
                    "latency_seconds": latency,
                    "rss_mb": rss,
                    "gpu_peak_mb": gpu_peak_mb,
                    "resumed": False,
                }
            )
            completed += 1
            progress.update(
                completed,
                suffix=f"origin={split}-{data.time_seconds[origin]} H={task.horizon.steps}",
                force=True,
            )
        progress.update(len(origins), suffix="done", force=True)
        records.sort(key=lambda item: item["origin"])
        atomic_write_csv(
            self._task_dir(task) / "predictions" / f"{split}-index.csv", records
        )
        latencies = np.asarray([item["latency_seconds"] for item in records], dtype=float)
        gpu_peaks = np.asarray([item["gpu_peak_mb"] for item in records], dtype=float)
        rss_values = np.asarray(
            [np.nan if item["rss_mb"] is None else item["rss_mb"] for item in records],
            dtype=float,
        )
        efficiency = {
            "n_origins": len(origins),
            "prediction_seconds_total": float(np.sum(latencies)),
            "origin_latency_mean_ms": float(np.mean(latencies) * 1000),
            "origin_latency_p50_ms": float(np.percentile(latencies, 50) * 1000),
            "origin_latency_p95_ms": float(np.percentile(latencies, 95) * 1000),
            "origin_latency_p99_ms": float(np.percentile(latencies, 99) * 1000),
            "workload_horizon_steps_per_second": float(
                len(origins) * data.workloads * task.horizon.steps / max(np.sum(latencies), 1e-9)
            ),
            "gpu_peak_mb_max": float(np.max(gpu_peaks)),
            "rss_mb_max": float(np.nanmax(rss_values)) if np.isfinite(rss_values).any() else None,
        }
        atomic_write_json(
            self._task_dir(task) / "predictions" / f"{split}-efficiency.json",
            efficiency,
        )
        return origins, efficiency

    def _load_predictions(
        self,
        data: PreparedData,
        task: TaskSpec,
        split: str,
        origins: Sequence[int],
    ) -> np.ndarray:
        arrays: list[np.ndarray] = []
        for origin in origins:
            path = self._prediction_path(task, split, origin)
            if not self._prediction_is_valid(path, task, data):
                raise ExperimentError(f"预测文件缺失或损坏: {path}")
            with np.load(path, allow_pickle=False) as archive:
                arrays.append(archive["predicted_membership"].astype(np.float32))
        return np.stack(arrays)

    def summarize_task(
        self,
        data: PreparedData,
        task: TaskSpec,
        origins_by_split: Mapping[str, Sequence[int]],
        efficiency_by_split: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        task_dir = self._task_dir(task)
        summaries: list[dict[str, Any]] = []
        for split in ("validation", "test"):
            all_origins = list(origins_by_split[split])
            predictions = self._load_predictions(data, task, split, all_origins)
            common_set = set(self._common_origins(data, split))
            common_indices = [index for index, origin in enumerate(all_origins) if origin in common_set]
            protocols: list[tuple[str, list[int], np.ndarray]] = [
                ("all_valid_origins", all_origins, predictions)
            ]
            if common_indices:
                protocols.append(
                    (
                        "common_origins",
                        [all_origins[index] for index in common_indices],
                        predictions[common_indices],
                    )
                )
            for protocol, origins, selected in protocols:
                summary = write_metric_bundle(
                    destination=task_dir / "metrics",
                    task=task,
                    split=split,
                    protocol=protocol,
                    origins=origins,
                    predictions=selected,
                    data=data,
                    calibration_bins=int(self.config.evaluation.get("calibration_bins", 15)),
                    bootstrap_samples=(
                        20 if self.smoke else int(self.config.evaluation.get("bootstrap_samples", 500))
                    ),
                    seed=stable_seed(self.config.seed, f"{task.task_id}:{split}:{protocol}"),
                    efficiency=efficiency_by_split.get(split),
                )
                summaries.append(summary)
        output = {"task": task.as_dict(), "summaries": summaries}
        atomic_write_json(task_dir / "metrics" / "summary.json", output)
        return output

    def run_task(self, data: PreparedData, task: TaskSpec, index: int, total: int) -> None:
        state = self._load_state()
        item = next(value for value in state["tasks"] if value["task_id"] == task.task_id)
        if item.get("status") == "complete" and (
            self._task_dir(task) / "metrics" / "summary.json"
        ).is_file():
            print_stage(f"[overall {index}/{total}] 跳过已完成任务 {task.task_id}")
            return
        print_stage(
            f"[overall {index}/{total}] 开始 {task.task_id} | "
            f"L={task.history_length} H={task.horizon.steps} direct"
        )
        try:
            self._update_task_state(
                task,
                status="training",
                started_at=item.get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                last_error=None,
            )
            model, model_metadata = self.train_task(data, task)
            self._update_task_state(task, status="evaluating", model_metadata=model_metadata)
            origins_by_split: dict[str, Sequence[int]] = {}
            efficiency_by_split: dict[str, Mapping[str, Any]] = {}
            for split in ("validation", "test"):
                origins, efficiency = self.evaluate_split(model, data, task, split)
                origins_by_split[split] = origins
                efficiency_by_split[split] = efficiency
            self._update_task_state(task, status="summarizing")
            print_stage(f"[{task.task_id}] 正在汇总逐h、逐容器、逐起点及校准指标")
            summary = self.summarize_task(
                data, task, origins_by_split, efficiency_by_split
            )
            self._update_task_state(
                task,
                status="complete",
                completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                summary=summary,
                last_error=None,
            )
            print_stage(f"[overall {index}/{total}] 完成 {task.task_id}")
        except Exception as error:
            self._update_task_state(
                task,
                status="failed",
                failed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                last_error=f"{type(error).__name__}: {error}",
            )
            raise

    def consolidate(self, tasks: Sequence[TaskSpec]) -> dict[str, Any]:
        combined_rows: list[dict[str, Any]] = []
        plot_rows: list[dict[str, Any]] = []
        for task in tasks:
            metrics_dir = self._task_dir(task) / "metrics"
            for path in sorted(metrics_dir.glob("*-metrics_by_h.csv")):
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        combined_rows.append(row)
                        if row.get("split") == "test":
                            plot_rows.append(
                                {
                                    "task_id": row.get("task_id"),
                                    "history_length": row.get("history_length"),
                                    "horizon_name": row.get("horizon_name"),
                                    "origin_protocol": row.get("origin_protocol"),
                                    "h": row.get("h"),
                                    "display_x": row.get("display_x"),
                                    "display_unit": row.get("display_unit"),
                                    "accuracy": row.get("classification.accuracy"),
                                    "accuracy_ci95_low": row.get("ci95.accuracy_low"),
                                    "accuracy_ci95_high": row.get("ci95.accuracy_high"),
                                    "membership_mae": row.get("membership.mae_mu"),
                                    "membership_js": row.get("membership.js_divergence"),
                                    "resource_mae": row.get("resources.overall.mae"),
                                    "n_samples": row.get("n_samples"),
                                }
                            )
        metrics_dir = self.run_dir / "metrics"
        atomic_write_csv(metrics_dir / "metrics_all_tasks.csv", combined_rows)
        atomic_write_csv(metrics_dir / "plot_data.csv", plot_rows)
        state = self._load_state()
        complete = sum(item["status"] == "complete" for item in state["tasks"])
        report = {
            "run_id": self.run_id,
            "mode": "smoke" if self.smoke else "formal",
            "tasks_complete": complete,
            "tasks_total": len(tasks),
            "all_complete": complete == len(tasks),
            "metrics_all_tasks": metrics_dir / "metrics_all_tasks.csv",
            "plot_data": metrics_dir / "plot_data.csv",
        }
        atomic_write_json(metrics_dir / "summary.json", report)
        return report

    def run_all(self) -> dict[str, Any]:
        tasks = self.task_queue()
        initial_state = self.ensure_run(tasks)
        data = prepare_data(self.config)
        self._cache_signature = data.cache_signature
        atomic_write_json(
            self.run_dir / "data_manifest.json",
            {
                "cache_path": data.cache_path,
                "cache_signature": data.cache_signature,
                "workloads": data.workloads,
                "steps": data.steps,
                "resources": list(map(str, data.resource_names)),
                "num_states": data.num_states,
                "train_end": data.train_end,
                "validation_end": data.validation_end,
                "membership_independent_of_L_and_H": True,
            },
        )
        overall = ProgressBar("overall", len(tasks), interval_seconds=self.interval)
        already_complete = sum(
            item.get("status") == "complete" for item in initial_state.get("tasks", [])
        )
        if already_complete:
            overall.update(already_complete, suffix="resume-scan", force=True)
            overall.finish()
        reported_complete = already_complete
        for index, task in enumerate(tasks, start=1):
            self.run_task(data, task, index, len(tasks))
            current_state = self._load_state()
            completed = sum(
                item.get("status") == "complete" for item in current_state.get("tasks", [])
            )
            if completed != reported_complete:
                overall.update(completed, suffix=f"last={task.task_id}", force=True)
                overall.finish()
                reported_complete = completed
        return self.consolidate(tasks)

    def summarize_existing(self) -> dict[str, Any]:
        tasks = self.task_queue()
        self.ensure_run(tasks)
        data = prepare_data(self.config)
        self._cache_signature = data.cache_signature
        for task in tasks:
            origins_by_split: dict[str, Sequence[int]] = {}
            efficiency_by_split: dict[str, Mapping[str, Any]] = {}
            for split in ("validation", "test"):
                origins = self._available_origins(data, task, split)
                if self.smoke:
                    origins = origins[:1]
                origins_by_split[split] = origins
                efficiency_path = self._task_dir(task) / "predictions" / f"{split}-efficiency.json"
                efficiency_by_split[split] = (
                    json.loads(efficiency_path.read_text(encoding="utf-8"))
                    if efficiency_path.is_file()
                    else {}
                )
            self.summarize_task(data, task, origins_by_split, efficiency_by_split)
            self._update_task_state(task, status="complete", last_error=None)
        return self.consolidate(tasks)

    def verify(self) -> dict[str, Any]:
        tasks = self.task_queue()
        self.ensure_run(tasks)
        data = prepare_data(self.config)
        checks: list[dict[str, Any]] = []
        for task in tasks:
            model_ok = (self._task_dir(task) / "models" / "model.pt").is_file()
            summary_ok = (self._task_dir(task) / "metrics" / "summary.json").is_file()
            prediction_ok = True
            prediction_count = 0
            for split in ("validation", "test"):
                origins = self._available_origins(data, task, split)
                if self.smoke:
                    origins = origins[:1]
                for origin in origins:
                    prediction_count += 1
                    prediction_ok &= self._prediction_is_valid(
                        self._prediction_path(task, split, origin), task, data
                    )
            checks.append(
                {
                    "task_id": task.task_id,
                    "model_ok": model_ok,
                    "predictions_ok": bool(prediction_ok),
                    "prediction_files_expected": prediction_count,
                    "metrics_ok": summary_ok,
                    "ok": bool(model_ok and prediction_ok and summary_ok),
                }
            )
        report = {
            "run_id": self.run_id,
            "ok": all(item["ok"] for item in checks),
            "tasks_ok": sum(item["ok"] for item in checks),
            "tasks_total": len(checks),
            "checks": checks,
        }
        atomic_write_json(self.run_dir / "verification.json", report)
        return report


__all__ = ["ExperimentError", "ExperimentRunner"]
