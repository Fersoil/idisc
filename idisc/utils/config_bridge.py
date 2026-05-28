"""Validate and resolve the Hydra-composed config into a plain dict.

Schema (enforced here):
  run.task          in {"train", "eval"}
  run.dataset_mode  in {"image", "video"}
  method.idr_source in {"afp", "sam3"}
  method.sam_mode   in {None, "replace", "translate"}; must be None iff idr_source == "afp"
  method.prompt.mode in {None, "multiclass", "singleclass"};
                      must be non-None iff a SAM3 pixel_encoder is configured;
                      when non-None, method.prompt.classes must be non-empty.

The single source of truth for prompt_mode / prompt_classes is method.prompt;
those values are mirrored into model.pixel_encoder so the encoder factory
keeps reading from its own config block.
"""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


_VALID_TASKS = {"train", "eval"}
_VALID_DATASET_MODES = {"image", "video"}
_VALID_IDR_SOURCES = {"afp", "sam3"}
_VALID_SAM_MODES = {None, "replace", "translate"}
_VALID_PROMPT_MODES = {None, "multiclass", "singleclass"}


def _to_container(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    else:
        resolved = OmegaConf.to_container(
            OmegaConf.create(cfg), resolve=True, throw_on_missing=True
        )
    resolved.pop("hydra", None)
    return resolved


def _validate(runtime: dict[str, Any]) -> None:
    task = runtime.get("run", {}).get("task")
    if task not in _VALID_TASKS:
        raise ValueError(f"run.task must be one of {_VALID_TASKS}, got {task!r}")

    dataset_mode = runtime.get("run", {}).get("dataset_mode")
    if dataset_mode not in _VALID_DATASET_MODES:
        raise ValueError(
            f"run.dataset_mode must be one of {_VALID_DATASET_MODES}, got {dataset_mode!r}"
        )

    method = runtime.get("method", {})
    idr_source = method.get("idr_source")
    sam_mode = method.get("sam_mode")
    if idr_source not in _VALID_IDR_SOURCES:
        raise ValueError(
            f"method.idr_source must be one of {_VALID_IDR_SOURCES}, got {idr_source!r}"
        )
    if sam_mode not in _VALID_SAM_MODES:
        raise ValueError(
            f"method.sam_mode must be one of {_VALID_SAM_MODES}, got {sam_mode!r}"
        )
    if idr_source == "afp" and sam_mode is not None:
        raise ValueError("method.sam_mode must be null when idr_source='afp'")
    if idr_source == "sam3" and sam_mode is None:
        raise ValueError(
            "method.sam_mode must be 'replace' or 'translate' when idr_source='sam3'"
        )

    encoder_name = (
        runtime.get("model", {}).get("pixel_encoder", {}).get("name", "")
    )
    is_sam3_encoder = encoder_name in ("sam3_image", "sam3_video")
    if (idr_source == "sam3") != is_sam3_encoder:
        raise ValueError(
            f"idr_source={idr_source!r} is inconsistent with pixel_encoder.name={encoder_name!r}"
        )

    prompt = method.get("prompt", {}) or {}
    prompt_mode = prompt.get("mode")
    if prompt_mode not in _VALID_PROMPT_MODES:
        raise ValueError(
            f"method.prompt.mode must be one of {_VALID_PROMPT_MODES}, got {prompt_mode!r}"
        )
    if is_sam3_encoder and prompt_mode is None:
        raise ValueError("SAM3 encoders require method.prompt.mode to be set")
    if prompt_mode is not None and not prompt.get("classes"):
        raise ValueError(
            f"method.prompt.mode={prompt_mode!r} requires non-empty method.prompt.classes"
        )


def build_runtime_config(
    cfg: DictConfig | dict[str, Any], repo_root: str | Path | None = None
) -> dict[str, Any]:
    runtime = _to_container(cfg)
    _validate(runtime)

    method = runtime["method"]
    prompt = method.get("prompt", {}) or {}
    prompt_mode = prompt.get("mode")
    prompt_classes = list(prompt.get("classes") or [])

    # Mirror prompt config into pixel_encoder so the encoder factory finds it.
    encoder_cfg = runtime.setdefault("model", {}).setdefault("pixel_encoder", {})
    if encoder_cfg.get("name", "").startswith("sam3"):
        encoder_cfg["prompt_mode"] = prompt_mode
        encoder_cfg["prompt_classes"] = prompt_classes

    # Convenience top-level pointer used by callers that don't want to dig.
    runtime["prompt_mode"] = prompt_mode

    return runtime


def save_resolved_config(runtime_cfg: dict[str, Any], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(runtime_cfg), f=str(output_path), resolve=True)
