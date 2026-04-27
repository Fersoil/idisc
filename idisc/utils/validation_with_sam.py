import json
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.utils.data.distributed
from torch import nn
from torch.utils.data import DataLoader

from idisc.utils.metrics import RunningMetric
from idisc.utils.misc import is_main_process

from sam3.model.sam3_image_processor import Sam3Processor

DEFAULT_SAM_PROMPT = "car . truck . person . bicycle . building . tree . road . sign . pole . barrier"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_imagenet(img_tensor):
    """Reverse ImageNet normalization to uint8 [0, 255] for SAM3 input."""
    mean = IMAGENET_MEAN.to(img_tensor.device)
    std = IMAGENET_STD.to(img_tensor.device)
    img = img_tensor * std + mean
    return (img * 255).clamp(0, 255).byte()


def log_losses(losses_all):
    for loss_name, loss_val in losses_all.items():
        print(f"Test/{loss_name}: ", loss_val)


def update_best(metrics_all, metrics_best="abs_rel"):
    curr_loss = []
    for metrics_name, metrics_value in metrics_all.items():
        if metrics_best in metrics_name:
            curr_loss.append(metrics_value)

    curr_loss = np.mean(curr_loss)
    if curr_loss < validate_with_sam.best_loss:
        validate_with_sam.best_loss = curr_loss
        validate_with_sam.best_metrics = metrics_all

    for metrics_name, metrics_value in metrics_all.items():
        try:
            print(
                f"{metrics_name} {round(validate_with_sam.best_metrics[metrics_name], 4)} ({round(metrics_value, 4)})"
            )
        except:
            print(f"Error in best. {metrics_name} ({round(metrics_value, 4)})")


def save_model(
    metrics_all, state_dict, run_save_dir, step, config, metrics_best="abs_rel"
):
    curr_loss = []
    curr_dataset = config["data"]["train_dataset"]
    for metrics_name, metrics_value in metrics_all.items():
        if metrics_best in metrics_name:
            curr_loss.append(metrics_value)
    curr_loss = np.mean(curr_loss)

    if curr_loss == validate_with_sam.best_loss:
        if step > 15000:
            try:
                torch.save(
                    state_dict, os.path.join(run_save_dir, f"{curr_dataset}-best.pt")
                )
                with open(
                    os.path.join(run_save_dir, f"{curr_dataset}-config.json"), "w+"
                ) as fp:
                    json.dump(config, fp)
            except OSError as e:
                print(f"Error while saving model: {e}")
            except:
                print("Generic error while saving")


def extract_instance_queries(processor, img_tensor, prompt=DEFAULT_SAM_PROMPT):
    """Run SAM3 on a single image and return filtered instance queries."""
    raw_img = denormalize_imagenet(img_tensor)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inference_state = processor.set_image(raw_img)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    instance_queries = output.get("instance_queries")
    if instance_queries is not None:
        instance_queries = instance_queries.float().clone()
    return instance_queries


def validate_with_sam(
    model_iDisc: nn.Module,
    model_SAM: nn.Module,
    test_loader: DataLoader,
    metrics_tracker: RunningMetric,
    context: torch.autocast,
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    save_dir: Optional[str] = None,
    step: int = 0,
    sam_prompt: str = DEFAULT_SAM_PROMPT,
):
    ds_losses = {}
    device = model_iDisc.device
    if save_dir is not None:
        run_save_dir = os.path.join(save_dir, run_id)
        os.makedirs(run_save_dir, exist_ok=True)

    processor = Sam3Processor(model_SAM)

    for i, batch in enumerate(test_loader):
        data = batch["image"].to(device)
        gt = batch["gt"].to(device)
        mask = batch["mask"].to(device)

        batch_preds = []
        batch_losses = {"opt": {}, "stat": {}}

        for img_idx in range(data.shape[0]):
            img_tensor = data[img_idx]
            instance_queries = extract_instance_queries(
                processor, img_tensor, prompt=sam_prompt
            )

            with context:
                pred, losses, _ = model_iDisc(
                    data[img_idx : img_idx + 1],
                    instance_queries=instance_queries,
                    gt=gt[img_idx : img_idx + 1] if gt is not None else None,
                    mask=mask[img_idx : img_idx + 1] if mask is not None else None,
                )
            batch_preds.append(pred)
            for loss_group in losses.values():
                for k, v in loss_group.items():
                    if k not in batch_losses["opt"]:
                        batch_losses["opt"][k] = v
                    else:
                        batch_losses["opt"][k] = batch_losses["opt"][k] + v

        preds = torch.cat(batch_preds, dim=0)
        losses_flat = {k: v / data.shape[0] for l in batch_losses.values() for k, v in l.items()}

        for loss_name, loss_val in losses_flat.items():
            ds_losses[loss_name] = (
                loss_val.detach().cpu().item() + i * ds_losses.get(loss_name, 0.0)
            ) / (i + 1)

        metrics_tracker.accumulate_metrics(
            gt.permute(0, 2, 3, 1), preds.permute(0, 2, 3, 1), mask.permute(0, 2, 3, 1)
        )

    losses_all = ds_losses
    metrics_all = metrics_tracker.get_metrics()
    metrics_tracker.reset_metrics()

    if is_main_process():
        log_losses(losses_all=losses_all)
        update_best(metrics_all=metrics_all, metrics_best="abs_rel")
        if save_dir is not None:
            with open(os.path.join(run_save_dir, f"metrics_{step}.json"), "w") as f:
                json.dump({**losses_all, **metrics_all}, f)
            save_model(
                metrics_all=metrics_all,
                state_dict=model_iDisc.state_dict(),
                config=config,
                metrics_best="abs_rel",
                run_save_dir=run_save_dir,
                step=step,
            )
