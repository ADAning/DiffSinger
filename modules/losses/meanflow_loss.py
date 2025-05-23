import torch
import torch.nn as nn
from torch import Tensor


class MeanFlowLoss(nn.Module):
    def __init__(self, loss_type, log_norm=True, adaptive_loss=True, gamma=0.0, c=1e-3):
        super().__init__()
        self.loss_type = loss_type
        # TODO: remove lognorm
        self.log_norm = False
        self.adaptive_loss = adaptive_loss
        if adaptive_loss:
            self.gamma = gamma
            self.c = c
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError()

    @staticmethod
    def _mask_non_padding(u_pred, u_tgt, non_padding=None):
        if non_padding is not None:
            non_padding = non_padding.transpose(1, 2).unsqueeze(1)
            return u_pred * non_padding, u_tgt * non_padding
        else:
            return u_pred, u_tgt

    @staticmethod
    def get_weights(t):
        eps = 1e-7
        t = t.float()
        t = torch.clip(t, 0 + eps, 1 - eps)
        weights = 0.398942 / t / (1 - t) * torch.exp(
            -0.5 * torch.log(t / (1 - t)) ** 2
        ) + eps
        return weights[:, None, None, None]

    def get_adaptive_weight(self, error):
        p = 1.0 - self.gamma
        w = 1.0 / (error + self.c).pow(p)
        return w

    def _forward(self, u_pred, u_tgt, t=None):
        error = self.loss(u_pred, u_tgt)
        weights = torch.tensor([1.], device=error.device)
        if self.log_norm:
            weights = weights * self.get_weights(t)
        if self.adaptive_loss:
            weights = weights * self.get_adaptive_weight(error)
        return weights.detach() * error

    def forward(self, u_pred: Tensor, u_tgt: Tensor, t: Tensor, non_padding: Tensor = None) -> Tensor:
        """
        :param u_pred: [B, 1, M, T]
        :param u_tgt: [B, 1, M, T]
        :param t: [B,]
        :param non_padding: [B, T, M]
        """
        u_pred, u_tgt = self._mask_non_padding(u_pred, u_tgt, non_padding)
        return self._forward(u_pred, u_tgt, t=t).mean()
