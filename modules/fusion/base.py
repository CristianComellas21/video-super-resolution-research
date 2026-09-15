from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from basicsr.archs.arch_util import flow_warp


class BaseFusion(nn.Module, ABC):
    """
    Contract:
      forward(prev, cur, nxt, flow_prev_to_cur, flow_next_to_cur) -> updated_cur

    prev/cur/nxt:        (B, F, H, W)
    flows (to cur):      (B, 2, H, W)   (0-flow => identity warp)
    output:              (B, F, H, W)
    """

    def __init__(self, feat_ch: int):
        super().__init__()
        self.feat_ch = int(feat_ch)

    @staticmethod
    def warp(feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        # flow_warp expects (B, H, W, 2)
        return flow_warp(feat, flow.permute(0, 2, 3, 1))

    def _check(self, prev, cur, nxt, fp2c, fn2c):
        # Útil durante desarrollo; si molesta, lo quitas o lo proteges con una flag.
        assert prev.dim() == 4 and cur.dim() == 4 and nxt.dim() == 4
        assert fp2c.dim() == 4 and fn2c.dim() == 4
        assert prev.shape == cur.shape == nxt.shape
        assert prev.size(1) == self.feat_ch
        assert fp2c.size(1) == 2 and fn2c.size(1) == 2
        assert fp2c.shape[0] == prev.shape[0] and fn2c.shape[0] == prev.shape[0]
        assert fp2c.shape[2:] == prev.shape[2:] and fn2c.shape[2:] == prev.shape[2:]

    # TODO: refactor fusions to change prev -> mem1, nxt -> mem2, becuase it is more generic (not necessarily prev/nxt but "memory from t-1/t+1") so we don't get confused when we apply recurrent fusion.
    @abstractmethod
    def forward(
        self,
        prev: torch.Tensor,
        cur: torch.Tensor,
        nxt: torch.Tensor,
        flow_cur_to_prev: torch.Tensor,
        flow_cur_to_next: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
