import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

IDR_SOURCE_AND_MODE_TO_VARIANT = {
    ("raw", "none"): "branch",
    ("raw", "concat"): "branch",
    ("raw", "replace"): "branch",
    ("afp", "none"): "baseline",
    ("afp", "replace"): "sam-replace",
    ("afp", "concat"): "sam-concat",
}


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
    runtime = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)

    method_cfg = runtime.get("method", {})
    idr_source = method_cfg.get("idr_source", "afp")
    sam_mode = method_cfg.get("sam_mode", "none")
    query_source = method_cfg.get("query_source", "online")

    variant_key = (idr_source, sam_mode)
    if variant_key not in IDR_SOURCE_AND_MODE_TO_VARIANT:
        raise ValueError(f"Unknown idr_source/sam_mode combination: {variant_key}")

    prompt_cfg = method_cfg.get("prompt", {})
    prompt_mode = prompt_cfg.get("mode")
    if prompt_mode is None:
        prompt_mode = method_cfg.get("prompt_mode", "none")

    runtime["variant"] = IDR_SOURCE_AND_MODE_TO_VARIANT[variant_key]
    runtime["prompt_mode"] = prompt_mode
    runtime["prompt"] = {
        "mode": prompt_mode,
        "classes": prompt_cfg.get("classes", []),
        "use_bbox": prompt_cfg.get("use_bbox", False),
        "strategy": prompt_cfg.get("strategy"),
    }

    # Keep method.prompt_mode aligned for any downstream code that reads it.
    runtime.setdefault("method", {})
    runtime["method"]["prompt_mode"] = prompt_mode

    if query_source == "cached":
        runtime["sam3_cache_dir"] = runtime.get("paths", {}).get("sam3_cache_dir")
        runtime["variant"] = "sam-cached-video"
    else:
        runtime["sam3_cache_dir"] = None

    return runtime


def save_resolved_config(runtime_cfg: dict[str, Any], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(runtime_cfg), f=str(output_path), resolve=True)
