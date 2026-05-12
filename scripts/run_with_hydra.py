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
        "dataset_name": cfg.get("dataset", {}).get("dataset_name"),
        "experiment_variant": cfg.get("variant"),
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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_dict(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Flatten nested dictionaries into slash-delimited summary keys."""
    flat: dict[str, Any] = {}

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                next_path = f"{path}/{key}" if path else str(key)
                _walk(value, next_path)
            return
        flat[f"{prefix}/{path}"] = node

    _walk(payload, "")
    return flat


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

    save_resolved_config(runtime_cfg, run_dir / "resolved_config.yaml")

    manifest = _build_manifest(runtime_cfg, run_dir, timestamp, git_branch, git_commit)
    _write_json(run_dir / "manifest.json", manifest)
    _write_stdout_log(run_dir / "stdout.log", manifest)

    with tracked_run(runtime_cfg, run_dir):
        task = runtime_cfg.get("run", {}).get("task", "eval")
        if task == "eval":
            from scripts.experiments.eval_depth import run_eval

            eval_cfg = {
                "variant": runtime_cfg["variant"],
                "prompt_mode": runtime_cfg.get("prompt_mode", "multiclass"),
                "model_file": _resolve_path(runtime_cfg["paths"]["pretrained_model"]),
                "base_path": _resolve_path(runtime_cfg["paths"]["base_path"]),
                "sam_checkpoint": _resolve_path(runtime_cfg["paths"].get("sam_checkpoint")),
                "output_dir": str(run_dir),
                "config": runtime_cfg,
                "load_pretrained": runtime_cfg.get("run", {}).get("load_pretrained", True),
            }
            metrics = run_eval(eval_cfg)
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                metrics_payload = _read_json(metrics_path)
            else:
                metrics_payload = {
                    "variant": runtime_cfg["variant"],
                    "prompt_mode": runtime_cfg.get("prompt_mode", "multiclass"),
                    "metrics": metrics,
                }
                _write_json(metrics_path, metrics_payload)

            log_payload: dict[str, Any] = {f"eval/{k}": v for k, v in metrics.items()}
            log_payload["meta/git_branch"] = git_branch
            log_payload["meta/git_commit"] = git_commit
            log_metrics(log_payload)

            summary_payload: dict[str, Any] = {
                "eval/variant": metrics_payload.get("variant"),
                "eval/prompt_mode": metrics_payload.get("prompt_mode"),
                "eval/elapsed_s": metrics_payload.get("elapsed_s"),
            }
            for key, value in metrics_payload.get("metrics", {}).items():
                summary_payload[f"eval/metrics/{key}"] = value
            summary_payload["meta/git_branch"] = git_branch
            summary_payload["meta/git_commit"] = git_commit
            log_summary(summary_payload)

            log_artifact(
                run_dir,
                name=f"{exp_id}-run-output",
                artifact_type="run-output",
                metadata={"task": task, "exp_id": exp_id},
            )

        elif task == "finetune":
            from scripts.experiments.finetune import run_finetune

            finetune_output = _resolve_path(runtime_cfg["finetune"]["output_dir"])
            finetune_cfg = {
                "variant": runtime_cfg["variant"],
                "prompt_mode": runtime_cfg.get("prompt_mode", "multiclass"),
                "model_file": _resolve_path(runtime_cfg["paths"]["pretrained_model"]),
                "base_path": _resolve_path(runtime_cfg["paths"]["base_path"]),
                "sam_checkpoint": _resolve_path(runtime_cfg["paths"].get("sam_checkpoint")),
                "use_sequence_dataset": runtime_cfg.get("finetune", {}).get("use_sequence_dataset", False),
                "config": runtime_cfg,
                "load_pretrained": runtime_cfg.get("run", {}).get("load_pretrained", True),
                "finetune": {
                    **runtime_cfg["finetune"],
                    "output_dir": finetune_output,
                },
            }
            summary = run_finetune(finetune_cfg)
            _write_json(run_dir / "metrics.json", summary)
            log_payload: dict[str, Any] = {
                "meta/git_branch": git_branch,
                "meta/git_commit": git_commit,
            }
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    log_payload[f"finetune/{key}"] = value
            log_metrics(log_payload)
            log_summary(log_payload)
            log_artifact(
                run_dir,
                name=f"{exp_id}-run-output",
                artifact_type="finetune-output",
                metadata={"task": task, "exp_id": exp_id},
            )

        else:
            raise ValueError(
                f"Unknown task: {task!r}. Valid values: eval | finetune"
            )

    print("Run complete.")
    print(f"Run dir: {run_dir}")
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
