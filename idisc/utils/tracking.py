from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

_ACTIVE_RUN = None
_ACTIVE_BACKEND = "none"


def _as_plain_dict(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


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
        name=run_cfg.get("name", run_cfg.get("exp_id")),
        tags=tags,
        dir=str(Path(run_dir).resolve()),
        config=cfg_dict,
    )

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


def finish_tracking() -> None:
    global _ACTIVE_RUN

    if _ACTIVE_BACKEND == "wandb" and _ACTIVE_RUN is not None:
        _ACTIVE_RUN.finish()
    _ACTIVE_RUN = None
