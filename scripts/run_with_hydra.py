#!/usr/bin/env python

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from idisc.utils.config_bridge import build_runtime_config, save_resolved_config
from idisc.utils.tracking import finish_tracking, init_tracking, log_metrics, log_summary
from scripts.experiments.eval_depth import run_eval


def _git_value(args: list[str], fallback: str = "unknown") -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return fallback


def _build_manifest(
    cfg: dict[str, Any],
    run_dir: Path,
    timestamp: str,
    git_branch: str,
    git_commit: str,
) -> dict[str, Any]:
    return {
        "exp_id": cfg.get("run", {}).get("exp_id"),
        "timestamp": timestamp,
        "git_branch": git_branch,
        "git_commit": git_commit,
        "dataset_name": cfg.get("dataset", {}).get("dataset_name"),
        "experiment_variant": cfg.get("method", {}).get("variant"),
        "legacy_config_path": cfg.get("dataset", {}).get("legacy_config_path"),
        "tracking_backend": cfg.get("tracking", {}).get("backend", "none"),
        "output_directory": str(run_dir),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_stdout_log(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "Hydra wrapper run metadata",
        f"exp_id={manifest['exp_id']}",
        f"timestamp={manifest['timestamp']}",
        f"git_branch={manifest['git_branch']}",
        f"git_commit={manifest['git_commit']}",
        f"dataset_name={manifest['dataset_name']}",
        f"experiment_variant={manifest['experiment_variant']}",
        f"legacy_config_path={manifest['legacy_config_path']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_id = cfg.run.exp_id
    git_branch = _git_value(["git", "branch", "--show-current"])
    git_commit = _git_value(["git", "rev-parse", "--short", "HEAD"])

    outputs_root = Path(cfg.paths.outputs_root)
    if not outputs_root.is_absolute():
        outputs_root = REPO_ROOT / outputs_root

    run_dir = outputs_root / f"{timestamp}_{exp_id}_{git_commit}"
    run_dir.mkdir(parents=True, exist_ok=True)

    runtime_cfg = build_runtime_config(cfg, repo_root=REPO_ROOT)
    runtime_cfg.setdefault("run", {})
    runtime_cfg["run"]["run_dir"] = str(run_dir)
    runtime_cfg["run"]["name"] = exp_id
    runtime_cfg["run"]["git_branch"] = git_branch
    runtime_cfg["run"]["git_commit"] = git_commit

    save_resolved_config(runtime_cfg, run_dir / "resolved_config.yaml")

    manifest = _build_manifest(runtime_cfg, run_dir, timestamp, git_branch, git_commit)
    _write_json(run_dir / "manifest.json", manifest)
    _write_stdout_log(run_dir / "stdout.log", manifest)

    init_tracking(runtime_cfg, run_dir)

    try:
        task = runtime_cfg.get("run", {}).get("task", "eval")
        if task != "eval":
            raise ValueError(f"Unsupported run.task in Stage 1: {task}")

        eval_cfg = {
            "variant": runtime_cfg["method"]["variant"],
            "prompt_mode": runtime_cfg["method"].get("prompt_mode", "multiclass"),
            "model_file": _resolve_path(runtime_cfg["paths"]["pretrained_model"]),
            "base_path": _resolve_path(runtime_cfg["paths"]["base_path"]),
            "sam_checkpoint": _resolve_path(runtime_cfg["paths"].get("sam_checkpoint")),
            "sam3_cache_dir": _resolve_path(runtime_cfg["paths"].get("sam3_cache_dir")),
            "output_dir": str(run_dir),
            "config": runtime_cfg,
        }

        metrics = run_eval(eval_cfg)
        _write_json(run_dir / "metrics.json", metrics)

        metrics_for_tracking = {f"eval/{k}": v for k, v in metrics.items()}
        metrics_for_tracking["meta/git_branch"] = git_branch
        metrics_for_tracking["meta/git_commit"] = git_commit
        log_metrics(metrics_for_tracking)
        log_summary(metrics_for_tracking)
    finally:
        finish_tracking()

    print("Run complete.")
    print(f"Run dir: {run_dir}")
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
