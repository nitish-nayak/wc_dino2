# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging

import torch
from torch import nn

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from dinov2.models.vision_transformer import BlockChunk
from dinov2.logging import DINODebugger

try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg, debug_cfg = None):
        super().__init__()
        self.cfg = cfg
        self.debug_cfg = debug_cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        logger.info("OPTIONS -- DINO")
        if self.do_dino or self.do_ibot:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

        # Add debugger for saving deeper-level of information (non-scalar)
        self.debugger = DINODebugger(debug_cfg, enabled = True) if self.debug_cfg is not None else None

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp, iteration=None):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = None
        if n_local_crops > 0:
            images["collated_local_crops"].cuda(non_blocking=True)

        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot:
                ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
                _dim = ibot_teacher_patch_tokens.shape[-1]
                if not self.ibot_separate_head:
                    buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                    buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                    torch.index_select(
                        ibot_teacher_patch_tokens.flatten(0, 1),
                        dim=0,
                        index=mask_indices_list,
                        out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                    )
                    teacher_patch_feats = buffer_tensor_teacher[n_cls_tokens:n_cls_tokens + n_masked_patches].detach()
                    tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                    teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                    masked_teacher_patch_tokens_after_head = tokens_after_head[
                        n_cls_tokens : n_cls_tokens + n_masked_patches
                    ]
                else:
                    buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                    torch.index_select(
                        ibot_teacher_patch_tokens.flatten(0, 1),
                        dim=0,
                        index=mask_indices_list,
                        out=buffer_tensor_teacher[:n_masked_patches],
                    )
                    teacher_patch_feats = buffer_tensor_teacher[:n_masked_patches].detach()
                    teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                    masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                        :n_masked_patches
                    ]
            # no ibot
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                teacher_patch_feats = None
                masked_teacher_ibot_softmaxed_centered = None

            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )

            else:
                raise NotImplementedError

            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered, teacher_patch_feats

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered, teacher_patch_feats = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        loss_dict = {}
        inputs_for_student_head_list = []

        loss_accumulator = 0  # for backprop
        #  student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
        #      [global_crops, local_crops], masks=[masks, None], is_training=True
        #  )
        student_global_backbone_output_dict = self.student.backbone(global_crops, masks=masks, is_training=True)
        if n_local_crops > 0:
            student_local_backbone_output_dict = self.student.backbone(local_crops, masks=None, is_training=True)
            # 1a: local crops cls tokens
            student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
            inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        student_patch_feats = None
        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            student_patch_feats = buffer_tensor_patch_tokens[:n_masked_patches].detach()
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]

        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        if n_local_crops > 0:
            student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            )["total_loss"] / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if self.debugger is not None:
            self.debugger.log_feature_stats(iteration, student_patch_feats, teacher_patch_feats)

        if do_dino:
            #  # compute loss
            #  dino_global_crops_loss_dict = self.dino_loss(
            #          student_output_list=[student_global_cls_tokens_after_head],
            #          teacher_out_softmaxed_centered_list=[
            #              teacher_dino_softmaxed_centered_list.flatten(0, 1)
            #          ],  # these were chunked and stacked in reverse so A is matched to B
            #      )
            #  dino_global_crops_losses = {k :
            #      v * loss_scales
            #      / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            #  for k, v in dino_global_crops_loss_dict.items()}
            #
            #  dino_global_crops_loss = dino_global_crops_losses["total_loss"]
            #  loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
            #  loss_dict.update({k : v for k, v in dino_global_crops_losses.items() if k != "total_loss"})
            #
            #  # accumulate loss
            #  loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

        if do_ibot:
            # compute loss
            ibot_patch_loss_dict = self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    #  masks_weight=masks_weight,
                    masks_weight=None,
                )
            ibot_patch_losses = {k : v*loss_scales*ibot_loss_scale for k,v in ibot_patch_loss_dict.items()}

            # store for display
            ibot_patch_loss = ibot_patch_losses["ibot"]
            loss_dict["ibot_loss"] = ibot_patch_loss / 2
            loss_dict.update({k : v/2 for k, v in ibot_patch_losses.items() if k != "ibot"})

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()

        return loss_dict


    def fsdp_synchronize_streams(self):
        if not self.need_to_synchronize_fsdp_streams:
            return

    # Try to flush device work; ignore if CUDA not available/initialized.
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        def _unwrap(m):
        # If wrapped by FSDP, unwrap to the real module; else return as-is.
            if m is None:
                return None
            return getattr(m, "_fsdp_wrapped_module", m)

    # Unwrap student/teacher backbones and heads
        student_bb = _unwrap(getattr(self.student, "backbone", None))
        teacher_bb = _unwrap(getattr(self.teacher, "backbone", None))
        student_head = _unwrap(getattr(self.student, "dino_head", None))
        teacher_head = _unwrap(getattr(self.teacher, "dino_head", None))

    # Copy teacher streams to student if the attribute exists
        try:
            t_streams_bb = getattr(teacher_bb, "_streams", None)
            if (t_streams_bb is not None) and (student_bb is not None):
                setattr(student_bb, "_streams", t_streams_bb)
        except Exception:
            pass

        try:
            t_streams_head = getattr(teacher_head, "_streams", None)
            if (t_streams_head is not None) and (student_head is not None):
                setattr(student_head, "_streams", t_streams_head)
        except Exception:
            pass

    # Done; never block training if nothing to sync
        self.need_to_synchronize_fsdp_streams = False


    def update_teacher(self, m):
        with torch.no_grad():
            student_param_list = []
            teacher_param_list = []

        # Try to collect params per submodule (backbone / heads), preferring FSDP modules if present
            for k in self.student.keys():
                s_mod = self.student[k]
                t_mod = self.teacher[k]

            # FSDP-aware path
                try:
                    s_fsdp = list(get_fsdp_modules(s_mod))
                    t_fsdp = list(get_fsdp_modules(t_mod))
                except Exception:
                    s_fsdp, t_fsdp = [], []

                if s_fsdp and t_fsdp and len(s_fsdp) == len(t_fsdp):
                # Some FSDP versions expose .params, otherwise fall back to .parameters()
                    for ms, mt in zip(s_fsdp, t_fsdp):
                        s_params = getattr(ms, "params", list(ms.parameters()))
                        t_params = getattr(mt, "params", list(mt.parameters()))
                        student_param_list += [p for p in s_params if p.requires_grad]
                        teacher_param_list += list(t_params)
                else:
                # Non-FSDP (or mismatch): just use the raw modules
                    student_param_list += [p for p in s_mod.parameters() if p.requires_grad]
                    teacher_param_list += list(t_mod.parameters())

        # Ultimate fallback: whole-model zip
            if not teacher_param_list:
                student_param_list = [p for p in self.student.parameters() if p.requires_grad]
                teacher_param_list = list(self.teacher.parameters())

        # Sanity: keep lists aligned
            assert len(student_param_list) == len(teacher_param_list), \
                f"Param length mismatch: student={len(student_param_list)} teacher={len(teacher_param_list)}"

        # Fast foreach when it works; otherwise per-tensor loop
            try:
                torch._foreach_mul_(teacher_param_list, m)
                torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)
            except Exception:
                for tp, sp in zip(teacher_param_list, student_param_list):
                    tp.mul_(m).add_(sp, alpha=1 - m)



    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        #  if has_batchnorms(self.student):
        #      raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
