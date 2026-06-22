from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from idisc.models._sam3_common import denormalize_imagenet


class Sam3TrackModule(nn.Module):
    def __init__(
        self,
        sam_checkpoint: Optional[str],
        prompt_classes: Sequence[str],
        confidence_threshold: float = 0.3,
        load_from_HF: Optional[bool] = None,
    ):
        super().__init__()
        from sam3.model_builder import build_sam3_video_model

        if load_from_HF is None:
            load_from_HF = sam_checkpoint is None
        self.vm = build_sam3_video_model(
            checkpoint_path=sam_checkpoint if not load_from_HF else None,
            load_from_HF=load_from_HF,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.vm.eval()
        self.vm.requires_grad_(False)
        self.vm.new_det_thresh = 0.0
        self.vm.score_threshold_detection = 0.0
        self.vm.hotstart_delay = 0
        self.vm.hotstart_unmatch_thresh = 0
        self.vm.hotstart_dup_thresh = 0
        self.prompt = " . ".join(prompt_classes)
        self.confidence_threshold = float(confidence_threshold)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vm.eval()
        return self

    @torch.no_grad()
    def forward(self, clip_norm: torch.Tensor) -> torch.Tensor:
        T, _, H, W = clip_norm.shape
        u8 = denormalize_imagenet(clip_norm)
        pil = [Image.fromarray(u8[t].permute(1, 2, 0).cpu().numpy()) for t in range(T)]

        labels = torch.zeros(T, H, W, dtype=torch.long, device=clip_norm.device)
        state = self.vm.init_state(resource_path=pil)
        self.vm.add_prompt(state, frame_idx=0, text_str=self.prompt)
        for fidx, out in self.vm.propagate_in_video(state):
            masks = out.get("out_binary_masks")
            ids = out.get("out_obj_ids")
            scores = out.get("out_probs")
            if masks is None or ids is None or len(ids) == 0:
                continue
            m = torch.as_tensor(np.asarray(masks)).float()
            if m.dim() == 4:
                m = m[:, 0]
            if m.shape[-2:] != (H, W):
                m = F.interpolate(m.unsqueeze(1), size=(H, W),
                                  mode="nearest").squeeze(1)
            ids_t = torch.as_tensor(np.asarray(ids)).long()
            sc = (torch.as_tensor(np.asarray(scores)).float()
                  if scores is not None else m.flatten(1).amax(1))
            for j in torch.argsort(sc).tolist():
                if sc[j] < self.confidence_threshold:
                    continue
                labels[fidx][m[j].to(clip_norm.device) > 0.5] = int(ids_t[j]) + 1
        del state
        return labels
