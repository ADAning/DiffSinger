from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from modules.backbones import build_backbone
from utils.hparams import hparams


class MeanFlow(nn.Module):
    def __init__(
        self,
        out_dims,
        num_feats=1,
        t_start=0.0,
        time_scale_factor=1000,
        backbone_type=None,
        backbone_args=None,
        spec_min=None,
        spec_max=None,
    ):
        super().__init__()
        self.velocity_fn: nn.Module = build_backbone(
            out_dims, num_feats, backbone_type, backbone_args
        )
        self.out_dims = out_dims
        self.num_feats = num_feats
        self.use_shallow_diffusion = hparams.get("use_shallow_diffusion", False)
        if self.use_shallow_diffusion:
            assert 0.0 <= t_start <= 1.0, "T_start should be in [0, 1]."
        else:
            t_start = 0.0
        self.t_start = t_start
        self.time_scale_factor = time_scale_factor

        # spec: [B, T, M] or [B, F, T, M]
        # spec_min and spec_max: [1, 1, M] or [1, 1, F, M] => transpose(-3, -2) => [1, 1, M] or [1, F, 1, M]
        spec_min = torch.FloatTensor(spec_min)[None, None, :out_dims].transpose(-3, -2)
        spec_max = torch.FloatTensor(spec_max)[None, None, :out_dims].transpose(-3, -2)
        self.register_buffer("spec_min", spec_min, persistent=False)
        self.register_buffer("spec_max", spec_max, persistent=False)
        # TODO: move mean / std -> hparams
        self.log_norm_mean = -0.4
        self.log_norm_std = 1.0
        self.rt_same_ratio = 0.0

    def sample_t_r(self, b, device):
        rand_time = torch.randn((b, 2), device=device) * self.log_norm_std + self.log_norm_mean
        normalized_time = rand_time.sigmoid()
        # normalized_time = torch.rand((b, 2), device=device)

        # max time -> t, min time -> r
        t = torch.max(normalized_time, dim=1).values
        r = torch.min(normalized_time, dim=1).values
        # TODO: t - r?
        t = self.t_start + (1.0 - self.t_start) * t
        r = self.t_start + (1.0 - self.t_start) * r
        
        # # TODO: r = t ratio?
        # mask = torch.rand((b, ), device=device) < self.rt_same_ratio
        # r[mask] = 0.0

        return t, r

    def p_losses(self, x_end, t, r, cond):
        x_start = torch.randn_like(x_end)
        x_t = x_start + t[:, None, None, None] * (x_end - x_start)
        
        scaled_t = t * self.time_scale_factor
        scaled_r = r * self.time_scale_factor
        velocity_fn = lambda x_t, scaled_t, scaled_r: self.velocity_fn(
            spec=x_t, diffusion_step=scaled_t, cond=cond, anchor_diffusion_step=scaled_r
        )
        v = x_end - x_start
        # Jacobian-vector product
        # also necessary to scale t of tangent vector simultaneously.
        u_pred, dudt = torch.func.jvp(
            velocity_fn,
            (x_t, scaled_t, scaled_r), 
            (v, torch.ones_like(t), torch.zeros_like(r)),
        )

        u_tgt = (v - ((t - r)[:, None, None, None]) * dudt).detach()

        return u_pred, u_tgt

    def forward(self, condition, gt_spec=None, src_spec=None, infer=True):
        cond = condition.transpose(1, 2)
        b, device = condition.shape[0], condition.device

        if not infer:
            # gt_spec: [B, T, M] or [B, F, T, M]
            spec = self.norm_spec(gt_spec).transpose(
                -2, -1
            )  # [B, M, T] or [B, F, M, T]
            if self.num_feats == 1:
                spec = spec[:, None, :, :]  # [B, F=1, M, T]
            t, r = self.sample_t_r(b, device=device)
            u_pred, u_tgt = self.p_losses(spec, t, r, cond=cond)
            return u_pred, u_tgt, t, r
        else:
            # src_spec: [B, T, M] or [B, F, T, M]
            if src_spec is not None:
                spec = self.norm_spec(src_spec).transpose(-2, -1)
                if self.num_feats == 1:
                    spec = spec[:, None, :, :]
            else:
                spec = None
            x = self.inference(cond, b=b, x_end=spec, device=device)
            return self.denorm_spec(x)

    @torch.no_grad()
    def sample_euler(self, x, t, r, dt, cond):
        x += self.velocity_fn(x, self.time_scale_factor * t, cond, self.time_scale_factor * r) * dt
        t += dt 
        return x, t
    
    @torch.no_grad()
    def inference(self, cond, b=1, x_end=None, device=None):
        noise = torch.randn(
            b, self.num_feats, self.out_dims, cond.shape[2], device=device
        )
        t_start = hparams.get("T_start_infer", self.t_start)
        if self.use_shallow_diffusion and t_start > 0:
            assert x_end is not None, "Missing shallow diffusion source."
            if t_start >= 1.0:
                t_start = 1.0
                x = x_end
            else:
                x = t_start * x_end + (1 - t_start) * noise
        else:
            t_start = 0.0
            x = noise

        algorithm = hparams["sampling_algorithm"]
        infer_step = hparams["sampling_steps"]

        if t_start < 1:
            dt = (1.0 - t_start) / max(1, infer_step)
            algorithm_fn = {
                "euler": self.sample_euler,
            }.get(algorithm)
            if algorithm_fn is None:
                raise ValueError(
                    f"Unsupported algorithm for Rectified Flow: {algorithm}."
                )
            dts = torch.tensor([dt]).to(x)
            t_end = torch.tensor([1.0]).to(x)
            for i in tqdm(
                range(infer_step),
                desc="sample time step",
                total=infer_step,
                disable=not hparams["infer"],
                leave=False,
            ):
                x, _ = algorithm_fn(x, t_end, t_start + i * dts, dt, cond)
            x = x.float()
        x = x.transpose(2, 3).squeeze(1)  # [B, F, M, T] => [B, T, M] or [B, F, T, M]
        return x

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


class RepetitiveMeanFlow(MeanFlow):
    def __init__(
        self,
        vmin: float | int | list,
        vmax: float | int | list,
        repeat_bins: int,
        time_scale_factor=1000,
        backbone_type=None,
        backbone_args=None,
    ):
        assert (
            isinstance(vmin, (float, int)) and isinstance(vmin, (float, int))
        ) or len(vmin) == len(vmax)
        num_feats = 1 if isinstance(vmin, (float, int)) else len(vmin)
        spec_min = [vmin] if num_feats == 1 else [[v] for v in vmin]
        spec_max = [vmax] if num_feats == 1 else [[v] for v in vmax]
        self.repeat_bins = repeat_bins
        super().__init__(
            out_dims=repeat_bins,
            num_feats=num_feats,
            time_scale_factor=time_scale_factor,
            backbone_type=backbone_type,
            backbone_args=backbone_args,
            spec_min=spec_min,
            spec_max=spec_max,
        )

    def norm_spec(self, x):
        """

        :param x: [B, T] or [B, F, T]
        :return [B, T, R] or [B, F, T, R]
        """
        if self.num_feats == 1:
            repeats = [1, 1, self.repeat_bins]
        else:
            repeats = [1, 1, 1, self.repeat_bins]
        return super().norm_spec(x.unsqueeze(-1).repeat(repeats))

    def denorm_spec(self, x):
        """

        :param x: [B, T, R] or [B, F, T, R]
        :return [B, T] or [B, F, T]
        """
        return super().denorm_spec(x).mean(dim=-1)


class PitchMeanFlow(RepetitiveMeanFlow):
    def __init__(
        self,
        vmin: float,
        vmax: float,
        cmin: float,
        cmax: float,
        repeat_bins,
        time_scale_factor=1000,
        backbone_type=None,
        backbone_args=None,
    ):
        self.vmin = vmin  # norm min
        self.vmax = vmax  # norm max
        self.cmin = cmin  # clip min
        self.cmax = cmax  # clip max
        super().__init__(
            vmin=vmin,
            vmax=vmax,
            repeat_bins=repeat_bins,
            time_scale_factor=time_scale_factor,
            backbone_type=backbone_type,
            backbone_args=backbone_args,
        )

    def norm_spec(self, x):
        return super().norm_spec(x.clamp(min=self.cmin, max=self.cmax))

    def denorm_spec(self, x):
        return super().denorm_spec(x).clamp(min=self.cmin, max=self.cmax)


class MultiVarianceMeanFlow(RepetitiveMeanFlow):
    def __init__(
        self,
        ranges: List[Tuple[float, float]],
        clamps: List[Tuple[float | None, float | None] | None],
        repeat_bins,
        time_scale_factor=1000,
        backbone_type=None,
        backbone_args=None,
    ):
        assert len(ranges) == len(clamps)
        self.clamps = clamps
        vmin = [r[0] for r in ranges]
        vmax = [r[1] for r in ranges]
        if len(vmin) == 1:
            vmin = vmin[0]
        if len(vmax) == 1:
            vmax = vmax[0]
        super().__init__(
            vmin=vmin,
            vmax=vmax,
            repeat_bins=repeat_bins,
            time_scale_factor=time_scale_factor,
            backbone_type=backbone_type,
            backbone_args=backbone_args,
        )

    def clamp_spec(self, xs: list | tuple):
        clamped = []
        for x, c in zip(xs, self.clamps):
            if c is None:
                clamped.append(x)
                continue
            clamped.append(x.clamp(min=c[0], max=c[1]))
        return clamped

    def norm_spec(self, xs: list | tuple):
        """

        :param xs: sequence of [B, T]
        :return: [B, F, T] => super().norm_spec(xs) => [B, F, T, R]
        """
        assert len(xs) == self.num_feats
        clamped = self.clamp_spec(xs)
        xs = torch.stack(clamped, dim=1)  # [B, F, T]
        if self.num_feats == 1:
            xs = xs.squeeze(1)  # [B, T]
        return super().norm_spec(xs)

    def denorm_spec(self, xs):
        """

        :param xs: [B, T, R] or [B, F, T, R] => super().denorm_spec(xs) => [B, T] or [B, F, T]
        :return: sequence of [B, T]
        """
        xs = super().denorm_spec(xs)
        if self.num_feats == 1:
            xs = [xs]
        else:
            xs = xs.unbind(dim=1)
        assert len(xs) == self.num_feats
        return self.clamp_spec(xs)
