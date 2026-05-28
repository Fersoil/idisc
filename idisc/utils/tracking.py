from __future__ import annotations

from pathlib import Path
from typing import Any, cast
import traceback

from omegaconf import OmegaConf

_ACTIVE_RUN = None
_ACTIVE_BACKEND = "none"


def _as_plain_dict(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    return cast(
        dict[str, Any],
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )


def init_tracking(cfg: Any, run_dir: str | Path):
    global _ACTIVE_BACKEND
    global _ACTIVE_RUN

    cfg_dict = _as_plain_dict(cfg)
    tracking_cfg = cfg_dict.get("tracking", {})
    enabled = bool(tracking_cfg.get("enabled", False))
    backend = tracking_cfg.get("backend", "none")

    _ACTIVE_BACKEND = backend if enabled else "none"
    if not enabled or backend == "none":
        _ACTIVE_RUN = None
        return None

    if backend != "wandb":
        raise ValueError(f"Unsupported tracking backend: {backend}")

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B backend requested but wandb is not installed. "
            "Install it with: pip install wandb"
        ) from exc

    run_cfg = cfg_dict.get("run", {})
    dataset_cfg = cfg_dict.get("dataset", {})
    method_cfg = cfg_dict.get("method", {})

    tags = []
    tags.extend(cfg_dict.get("tags", []))
    tags.extend(tracking_cfg.get("tags", []))

    _ACTIVE_RUN = wandb.init(
        project=tracking_cfg.get("project", "idisc"),
        entity=tracking_cfg.get("entity"),
        mode=tracking_cfg.get("mode", "online"),
        name=run_cfg.get("wandb_name") or run_cfg.get("name") or run_cfg.get("exp_id"),
        tags=tags,
        dir=str(Path(run_dir).resolve()),
        config=cfg_dict,
    )

    # Align metric axes across experiments by using a shared global step.
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")
    wandb.define_metric("eval/*", step_metric="global_step")

    _ACTIVE_RUN.summary["run.exp_id"] = run_cfg.get("exp_id")
    _ACTIVE_RUN.summary["dataset"] = dataset_cfg.get("dataset_name")
    _ACTIVE_RUN.summary["method.variant"] = method_cfg.get("variant")

    return _ACTIVE_RUN


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    if _ACTIVE_BACKEND != "wandb" or _ACTIVE_RUN is None:
        return

    payload = dict(metrics)
    if step is None:
        _ACTIVE_RUN.log(payload)
    else:
        _ACTIVE_RUN.log(payload, step=step)


def log_summary(metrics: dict[str, Any]) -> None:
    if _ACTIVE_BACKEND != "wandb" or _ACTIVE_RUN is None:
        return

    for key, value in metrics.items():
        _ACTIVE_RUN.summary[key] = value


def log_artifact(
    artifact_path: str | Path,
    name: str | None = None,
    artifact_type: str = "run-output",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log a file or directory as a W&B artifact for traceability."""
    if _ACTIVE_BACKEND != "wandb" or _ACTIVE_RUN is None:
        return

    import wandb

    path = Path(artifact_path)
    artifact_name = name or path.name
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=metadata or {})
    if path.is_dir():
        artifact.add_dir(str(path))
    else:
        artifact.add_file(str(path))
    _ACTIVE_RUN.log_artifact(artifact)


def log_error(error: Exception) -> None:
    """Log exception details to W&B (traceback + metadata)."""
    if _ACTIVE_BACKEND != "wandb" or _ACTIVE_RUN is None:
        return

    tb = traceback.format_exc()
    _ACTIVE_RUN.log(
        {
            "status": "failed",
            "error_type": type(error).__name__,
            "error_traceback": tb,
        }
    )
    _ACTIVE_RUN.summary.update(
        {
            "status": "failed",
            "error_type": type(error).__name__,
        }
    )


def finish_tracking(exit_code: int = 0, quiet: bool = False) -> None:
    """Finish W&B run with status. Best practice: non-zero exit_code → "Failed"."""
    global _ACTIVE_RUN
    if _ACTIVE_BACKEND == "wandb" and _ACTIVE_RUN is not None:
        _ACTIVE_RUN.finish(exit_code=exit_code, quiet=quiet)
    _ACTIVE_RUN = None
