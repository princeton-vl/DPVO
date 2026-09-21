"""Scatter ops implemented with native PyTorch only."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"Invalid dim={dim} for tensor with {ndim} dims")
    return dim


def _infer_dim_size(index: torch.Tensor, dim_size: Optional[int]) -> int:
    if dim_size is not None:
        return int(dim_size)
    if index.numel() == 0:
        return 0
    max_index = int(index.max().item())
    return max(0, max_index + 1)


def _broadcast_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    if index.dtype != torch.long:
        index = index.long()

    if index.dim() == 1:
        view_shape = [1] * src.dim()
        view_shape[dim] = index.numel()
        index = index.view(*view_shape)
    elif index.dim() != src.dim():
        raise ValueError(
            f"index must be 1D or match src dims (got index.dim={index.dim()}, src.dim={src.dim()})"
        )

    expand_shape = []
    for axis in range(src.dim()):
        idx_sz = index.size(axis)
        src_sz = src.size(axis)

        if axis == dim:
            if idx_sz != src_sz:
                raise ValueError(
                    f"index size mismatch at dim={dim}: expected {src_sz}, got {idx_sz}"
                )
            expand_shape.append(src_sz)
            continue

        if idx_sz not in (1, src_sz):
            raise ValueError(
                f"index not broadcastable at axis={axis}: expected 1 or {src_sz}, got {idx_sz}"
            )
        expand_shape.append(src_sz)

    return index.expand(*expand_shape)


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    dim = _normalize_dim(dim, src.dim())
    index = _broadcast_index(index, src, dim)
    dim_size = _infer_dim_size(index, dim_size)

    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

    if dim_size == 0 or src.numel() == 0:
        return out

    valid = (index >= 0) & (index < dim_size)
    safe_index = index.clamp(min=0, max=dim_size - 1)
    safe_src = torch.where(valid, src, torch.zeros_like(src))
    out.scatter_add_(dim, safe_index, safe_src)
    return out


def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = _normalize_dim(dim, src.dim())
    index = _broadcast_index(index, src, dim)
    dim_size = _infer_dim_size(index, dim_size)

    out_shape = list(src.shape)
    out_shape[dim] = dim_size

    if torch.is_floating_point(src):
        fill_value = float("-inf")
    elif src.dtype == torch.bool:
        fill_value = False
    else:
        fill_value = torch.iinfo(src.dtype).min

    out = torch.full(out_shape, fill_value, dtype=src.dtype, device=src.device)
    arg = torch.full(out_shape, -1, dtype=torch.long, device=src.device)

    if dim_size == 0 or src.numel() == 0:
        return out, arg

    valid = (index >= 0) & (index < dim_size)
    safe_index = index.clamp(min=0, max=dim_size - 1)
    fill = torch.full_like(src, fill_value)
    safe_src = torch.where(valid, src, fill)
    out.scatter_reduce_(dim, safe_index, safe_src, reduce="amax", include_self=True)

    # Reconstruct argmax positions with "first occurrence" tie-breaking.
    pos = torch.arange(src.size(dim), device=src.device, dtype=torch.long)
    pos_view = [1] * src.dim()
    pos_view[dim] = src.size(dim)
    pos = pos.view(*pos_view).expand_as(src)

    gathered_max = out.gather(dim, safe_index)
    is_winner = valid & (src == gathered_max)

    sentinel = torch.full_like(pos, src.size(dim))
    cand = torch.where(is_winner, pos, sentinel)

    arg = torch.full(out_shape, src.size(dim), dtype=torch.long, device=src.device)
    arg.scatter_reduce_(dim, safe_index, cand, reduce="amin", include_self=True)
    arg = torch.where(arg == src.size(dim), torch.full_like(arg, -1), arg)
    return out, arg


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise TypeError("scatter_softmax fallback expects floating-point src")

    dim = _normalize_dim(dim, src.dim())
    index_b = _broadcast_index(index, src, dim)
    dim_size = _infer_dim_size(index_b, dim_size)

    if dim_size == 0 or src.numel() == 0:
        return torch.zeros_like(src)

    safe_index = index_b.clamp(min=0, max=dim_size - 1)
    valid = (index_b >= 0) & (index_b < dim_size)

    max_per_group, _ = scatter_max(src, index_b, dim=dim, dim_size=dim_size)
    max_src = max_per_group.gather(dim, safe_index)

    shifted = torch.where(valid, src - max_src, torch.zeros_like(src))
    exp = torch.exp(shifted)
    exp = torch.where(valid, exp, torch.zeros_like(exp))

    denom = scatter_sum(exp, index_b, dim=dim, dim_size=dim_size)
    denom_src = denom.gather(dim, safe_index)

    eps = 1e-12 if src.dtype in (torch.float32, torch.float64) else 1e-4
    return exp / denom_src.clamp_min(eps)
