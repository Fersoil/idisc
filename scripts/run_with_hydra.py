#!/usr/bin/env python

import json
import subprocess
import sys
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))   # append, not insert — venv packages win
from idisc.utils.config_bridge import build_runtime_config, save_resolved_config
from idisc.utils.tracking import (
    finish_tracking,
    init_tracking,
    log_artifact,
    log_error,
    log_metrics,
    log_summary,
)


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
        "dataset_name": cfg.get("dataset_name"),
        "dataset_mode": cfg.get("run", {}).get("dataset_mode"),
        "idr_source": cfg.get("method", {}).get("idr_source"),
        "sam_mode": cfg.get("method", {}).get("sam_mode"),
        "tracking_backend": cfg.get("tracking", {}).get("backend", "none"),
        "output_directory": str(run_dir),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_stdout_log(path: Path, manifest: dict[str, Any]) -> None:
    lines = [f"{k}={v}" for k, v in manifest.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


@contextmanager
def tracked_run(cfg: dict[str, Any], run_dir: Path):
    """Clean W&B context: auto-logs errors, sets status, finishes correctly."""
    init_tracking(cfg, run_dir)
    try:
        yield
    except Exception as error:
        log_error(error)
        finish_tracking(exit_code=1, quiet=True)
        raise
    else:
        log_summary({"status": "success"})
        finish_tracking(exit_code=0)


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
    runtime_cfg["paths"] = {
        k: _resolve_path(v) if isinstance(v, str) else v
        for k, v in runtime_cfg.get("paths", {}).items()
    }

    save_resolved_config(runtime_cfg, run_dir / "resolved_config.yaml")

    manifest = _build_manifest(runtime_cfg, run_dir, timestamp, git_branch, git_commit)
    _write_json(run_dir / "manifest.json", manifest)
    _write_stdout_log(run_dir / "stdout.log", manifest)

    with tracked_run(runtime_cfg, run_dir):
        task = runtime_cfg["run"]["task"]
        if task == "train":
            from scripts.train import run_train
            summary = run_train(runtime_cfg)
        else:
            from scripts.experiments.eval_depth import run_eval
            metrics = run_eval(
                runtime_cfg,
                checkpoint_path=runtime_cfg["paths"]["pretrained_model"],
                output_dir=str(run_dir),
            )
            summary = {"task": "eval", **metrics}
        _write_json(run_dir / "metrics.json", summary)
        log_payload: dict[str, Any] = {
            "meta/git_branch": git_branch,
            "meta/git_commit": git_commit,
        }
        if task == "eval":
            log_payload["global_step"] = 0
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                log_payload[f"{task}/{key}"] = value
        log_metrics(log_payload)
        log_summary(log_payload)
        log_artifact(
            run_dir,
            name=f"{exp_id}-run-output",
            artifact_type=f"{task}-output",
            metadata={"exp_id": exp_id, "task": task},
        )

    print("Run complete.")
    print(f"Run dir: {run_dir}")
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
