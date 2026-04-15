import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _to_config(cfg: DictConfig | dict[str, Any]) -> DictConfig:
    if isinstance(cfg, DictConfig):
        return cfg
    return OmegaConf.create(cfg)


def _load_legacy_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_legacy_path(cfg: DictConfig, repo_root: Path | None = None) -> Path:
    legacy_rel = cfg.dataset.legacy_config_path
    candidate = Path(legacy_rel)
    if candidate.is_absolute():
        return candidate

    root = repo_root if repo_root is not None else Path.cwd()
    return (root / candidate).resolve()


def build_runtime_config(
    cfg: DictConfig | dict[str, Any], repo_root: str | Path | None = None
) -> dict[str, Any]:
    """Merge legacy JSON config with Hydra overlays and return a resolved dict.

    Precedence (last wins):
      1) legacy JSON base config
      2) Hydra composed config (including CLI overrides)
    """
    hydra_cfg = _to_config(cfg)
    repo_path = Path(repo_root).resolve() if repo_root is not None else None
    legacy_path = _resolve_legacy_path(hydra_cfg, repo_root=repo_path)

    legacy_cfg = OmegaConf.create(_load_legacy_json(legacy_path))

    overlay_cfg = OmegaConf.create(
        OmegaConf.to_container(hydra_cfg, resolve=True, throw_on_missing=True)
    )
    if "hydra" in overlay_cfg:
        del overlay_cfg["hydra"]

    merged = OmegaConf.merge(legacy_cfg, overlay_cfg)
    return OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)


def save_resolved_config(runtime_cfg: dict[str, Any], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(runtime_cfg), f=str(output_path), resolve=True)
