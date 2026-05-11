import os
import json
import zlib
import struct
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.ao.quantization import QConfig
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torchvision import datasets, transforms

from model import build_model
import math


# ============================================================
# User-fixed paths
# ============================================================
CKPT_PATH = r"E:/06.CoreCreation/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_best_fakequant.pth"
DATA_ROOT = r"E:/06.CoreCreation/06_RepOpt_VGG/data"
EXPORT_DIR = r"./hardware_package"
WIDTH_MULT = 1.0
RANDOM_SEED = 42


# ============================================================
# Fixed export spec
# ============================================================
ENDIAN = "<"
ALIGN_BYTES = 32
CFG_RECORD_BYTES = 128
QUANT_RECORD_BYTES = 128
TILE_RECORD_BYTES = 128
HEADER_BYTES = 64
FORMAT_VERSION = 1

MAGIC_LAYER_CFG = b"LYRCFG01"
MAGIC_QUANT_CFG = b"QNTCFG01"
MAGIC_WEIGHT = b"WEIGHT01"
MAGIC_BIAS = b"BIAS0001"
MAGIC_INPUT = b"INPUT001"
MAGIC_GOLDEN = b"GOLDEN01"
MAGIC_LAYER_OUT = b"LYROUT01"
MAGIC_TILE_PLAN = b"TILPLN01"

# op_type encoding
OP_QUANTIZE = 1
OP_CONV_RELU = 2
OP_MAXPOOL = 3
OP_AVGPOOL = 4
OP_FLATTEN = 5
OP_LINEAR = 6
OP_DEQUANTIZE = 7

DTYPE_NONE = 0
DTYPE_INT8 = 1
DTYPE_UINT8 = 2
DTYPE_INT32 = 3
DTYPE_FLOAT32 = 4

QSCHEME_NONE = 0
QSCHEME_PER_TENSOR_AFFINE = 1
QSCHEME_PER_CHANNEL_AFFINE = 2
QSCHEME_PER_TENSOR_SYMMETRIC = 3
QSCHEME_PER_CHANNEL_SYMMETRIC = 4


def avgpool_integer_same_qparams(
    x: np.ndarray,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
    out_dtype_code: int,
) -> np.ndarray:
    n, c, h, w = x.shape

    if k_h <= 0 or k_w <= 0 or s_h <= 0 or s_w <= 0:
        raise ValueError(f"Invalid avgpool params: k=({k_h},{k_w}), s=({s_h},{s_w}), input_shape={x.shape}")

    out_h = (h - k_h) // s_h + 1
    out_w = (w - k_w) // s_w + 1
    out = np.empty((n, c, out_h, out_w), dtype=np.int64)

    kernel_area = k_h * k_w
    for oh in range(out_h):
        hs = oh * s_h
        for ow in range(out_w):
            ws = ow * s_w
            patch = x[:, :, hs:hs + k_h, ws:ws + k_w].astype(np.int64)
            s = np.sum(patch, axis=(2, 3))
            out[:, :, oh, ow] = div_round_half_away_from_zero(s, kernel_area)

    out = clamp_to_dtype_range_np(out, out_dtype_code)
    if out_dtype_code == DTYPE_INT8:
        return out.astype(np.int8)
    if out_dtype_code == DTYPE_UINT8:
        return out.astype(np.uint8)
    raise RuntimeError(f"Unsupported out_dtype_code={out_dtype_code}")

def dequantize_int_tensor_with_qparams(
    x_int: np.ndarray,
    qparams: Dict[str, Any],
) -> torch.Tensor:
    if not qparams["is_quantized"]:
        raise RuntimeError("qparams says tensor is not quantized")

    if qparams["qscheme_code"] in (QSCHEME_PER_TENSOR_AFFINE, QSCHEME_PER_TENSOR_SYMMETRIC):
        scale = float(qparams["scale"])
        zp = int(qparams["zero_point"])
        y = (x_int.astype(np.float32) - np.float32(zp)) * np.float32(scale)
        return torch.from_numpy(y)

    if qparams["qscheme_code"] in (QSCHEME_PER_CHANNEL_AFFINE, QSCHEME_PER_CHANNEL_SYMMETRIC):
        scales = np.asarray(qparams["scales"], dtype=np.float32)
        zps = np.asarray(qparams["zero_points"], dtype=np.float32)
        axis = int(qparams["axis"])

        shape = [1] * x_int.ndim
        shape[axis] = scales.shape[0]

        scales_bc = scales.reshape(shape)
        zps_bc = zps.reshape(shape)

        y = (x_int.astype(np.float32) - zps_bc) * scales_bc
        return torch.from_numpy(y)

    raise RuntimeError(f"Unsupported qscheme_code: {qparams['qscheme_code']}")

def round_half_away_from_zero_scalar(x: float) -> int:
    if x >= 0:
        return int(math.floor(x + 0.5))
    return -int(math.floor(-x + 0.5))


def div_round_half_away_from_zero(numer: np.ndarray, denom: int) -> np.ndarray:
    if denom <= 0:
        raise ValueError(f"denom must be > 0, got {denom}")
    numer = numer.astype(np.int64, copy=False)
    sign = np.where(numer >= 0, 1, -1)
    absn = np.abs(numer)
    out = (absn + denom // 2) // denom
    out = out * sign
    return out.astype(np.int64)


def clamp_to_dtype_range_np(x: np.ndarray, dtype_code: int) -> np.ndarray:
    if dtype_code == DTYPE_INT8:
        return np.clip(x, -128, 127)
    if dtype_code == DTYPE_UINT8:
        return np.clip(x, 0, 255)
    if dtype_code == DTYPE_INT32:
        return np.clip(x, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    raise ValueError(f"Clamp not defined for dtype_code={dtype_code}")


def quantize_multiplier_smaller_than_one(real_scale: float) -> Tuple[int, int]:
    if real_scale <= 0.0 or real_scale >= 1.0:
        raise ValueError(f"This helper expects 0 < real_scale < 1. Got {real_scale}")

    shift = 0
    while real_scale < 0.5:
        real_scale *= 2.0
        shift += 1

    q = round_half_away_from_zero_scalar(real_scale * (1 << 31))
    if q == (1 << 31):
        q //= 2
        shift -= 1
    return q, shift


def saturating_rounding_doubling_high_mul(a: np.ndarray, multiplier: int) -> np.ndarray:
    a64 = a.astype(np.int64, copy=False)
    b64 = np.int64(multiplier)
    ab = a64 * b64
    nudge = np.where(ab >= 0, np.int64(1 << 30), np.int64(1) - np.int64(1 << 30))
    return ((ab + nudge) // np.int64(1 << 31)).astype(np.int64)


def rounding_divide_by_pot(x: np.ndarray, exponent: int) -> np.ndarray:
    if exponent < 0:
        raise ValueError(f"exponent must be >= 0, got {exponent}")
    x64 = x.astype(np.int64, copy=False)
    if exponent == 0:
        return x64
    mask = np.int64((1 << exponent) - 1)
    remainder = np.bitwise_and(x64, mask)
    threshold = (mask >> 1) + np.where(x64 < 0, np.int64(1), np.int64(0))
    return np.right_shift(x64, exponent) + (remainder > threshold).astype(np.int64)


def requantize_per_channel(
    acc: np.ndarray,
    input_scale: float,
    weight_scales: np.ndarray,
    output_scale: float,
    output_zero_point: int,
    out_dtype_code: int,
) -> np.ndarray:
    oc = acc.shape[1]
    if weight_scales.shape[0] != oc:
        raise RuntimeError(f"weight_scales len mismatch: got={weight_scales.shape[0]}, expected out_channels={oc}")

    out = np.empty_like(acc, dtype=np.int8 if out_dtype_code == DTYPE_INT8 else np.uint8)

    for c in range(oc):
        rs = (input_scale * float(weight_scales[c])) / output_scale
        if not (0.0 < rs < 1.0):
            raise RuntimeError(f"real_scale out of expected range (0,1): channel {c}, value={rs}")

        m, s = quantize_multiplier_smaller_than_one(rs)
        cur = acc[:, c].astype(np.int64)
        scaled = saturating_rounding_doubling_high_mul(cur, m)
        scaled = rounding_divide_by_pot(scaled, s)
        scaled = scaled + np.int64(output_zero_point)
        scaled = clamp_to_dtype_range_np(scaled, out_dtype_code)

        if out_dtype_code == DTYPE_INT8:
            out[:, c] = scaled.astype(np.int8)
        elif out_dtype_code == DTYPE_UINT8:
            out[:, c] = scaled.astype(np.uint8)
        else:
            raise RuntimeError(f"Unsupported out_dtype_code={out_dtype_code}")

    return out

def pad_nchw_int(x: np.ndarray, p_h: int, p_w: int, pad_value: int) -> np.ndarray:
    if p_h == 0 and p_w == 0:
        return x
    return np.pad(
        x,
        ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=pad_value,
    )


def im2col_nchw(
    x: np.ndarray,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
    p_h: int,
    p_w: int,
    input_zero_point: int,
) -> np.ndarray:
    n, c, h, w = x.shape
    xpad = pad_nchw_int(x, p_h, p_w, input_zero_point)
    hp, wp = xpad.shape[2], xpad.shape[3]
    out_h = (hp - k_h) // s_h + 1
    out_w = (wp - k_w) // s_w + 1

    cols = np.empty((n, out_h * out_w, c * k_h * k_w), dtype=np.int32)
    idx = 0
    for oh in range(out_h):
        hs = oh * s_h
        for ow in range(out_w):
            ws = ow * s_w
            patch = xpad[:, :, hs:hs + k_h, ws:ws + k_w]
            cols[:, idx, :] = patch.reshape(n, -1).astype(np.int32)
            idx += 1
    return cols


def conv2d_integer_im2col(
    x_q: np.ndarray,
    w_q: np.ndarray,
    b_int32: np.ndarray,
    input_zero_point: int,
    weight_zero_points: np.ndarray,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
) -> np.ndarray:
    n, ic, h, w = x_q.shape
    oc, ic2, kh, kw = w_q.shape
    if ic != ic2:
        raise RuntimeError(f"Conv input channels mismatch: x={ic}, w={ic2}")

    cols = im2col_nchw(x_q, kh, kw, stride_h, stride_w, pad_h, pad_w, input_zero_point)
    xpad_h = h + 2 * pad_h
    xpad_w = w + 2 * pad_w
    out_h = (xpad_h - kh) // stride_h + 1
    out_w = (xpad_w - kw) // stride_w + 1

    cols = cols - np.int32(input_zero_point)
    w_flat = w_q.reshape(oc, -1).astype(np.int32)

    acc = np.empty((n, oc, out_h * out_w), dtype=np.int32)
    for c_out in range(oc):
        wzp = int(weight_zero_points[c_out])
        wf = w_flat[c_out] - np.int32(wzp)
        acc[:, c_out, :] = cols @ wf + np.int32(b_int32[c_out])

    return acc.reshape(n, oc, out_h, out_w)


def linear_integer(
    x_q: np.ndarray,
    w_q: np.ndarray,
    b_int32: np.ndarray,
    input_zero_point: int,
    weight_zero_points: np.ndarray,
) -> np.ndarray:
    n, ic = x_q.shape
    oc, ic2 = w_q.shape
    if ic != ic2:
        raise RuntimeError(f"Linear input dim mismatch: x={ic}, w={ic2}")

    x_center = x_q.astype(np.int32) - np.int32(input_zero_point)
    out = np.empty((n, oc), dtype=np.int32)
    for c_out in range(oc):
        wf = w_q[c_out].astype(np.int32) - np.int32(weight_zero_points[c_out])
        out[:, c_out] = x_center @ wf + np.int32(b_int32[c_out])
    return out


def relu_quantized_np(x: np.ndarray, zero_point: int, out_dtype_code: int) -> np.ndarray:
    x64 = x.astype(np.int64, copy=False)
    x64 = np.maximum(x64, np.int64(zero_point))
    x64 = clamp_to_dtype_range_np(x64, out_dtype_code)
    if out_dtype_code == DTYPE_INT8:
        return x64.astype(np.int8)
    if out_dtype_code == DTYPE_UINT8:
        return x64.astype(np.uint8)
    raise RuntimeError(f"Unsupported out_dtype_code={out_dtype_code}")

# ============================================================
# Model reconstruction
# ============================================================
class QuantRepOptVGGLike(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = build_model(num_classes=num_classes, width_mult=width_mult)
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def fuse_model(model: nn.Module):
    """
    融合 Conv + BN + ReLU
    注意：融合前必须 eval()
    """
    for stage_name in ["stage1", "stage2", "stage3", "stage4"]:
        stage = getattr(model.model, stage_name)
        for m in stage:
            if hasattr(m, "conv") and hasattr(m, "bn") and hasattr(m, "relu"):
                torch.ao.quantization.fuse_modules(
                    m,
                    [["conv", "bn", "relu"]],
                    inplace=True,
                )


def get_qat_qconfig_int8_int8():
    """
    激活 int8，权重 int8
    推荐给你当前 NPU 流程：
        activation: per-tensor symmetric int8
        weight    : per-channel symmetric int8
    """
    activation_fake_quant = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )

    weight_fake_quant = FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        reduce_range=False,
    )

    return QConfig(
        activation=activation_fake_quant,
        weight=weight_fake_quant,
    )


def build_qat_fakequant_model(width_mult=1.0):
    """
    这里不做 convert()，只构建 prepare_qat 后的 fake-quant 模型。
    后续导出全部基于 fake-quant 模型的 scale / zero_point / float权重手动量化。
    """
    model = QuantRepOptVGGLike(num_classes=10, width_mult=width_mult)
    model.eval()
    model.cpu()
    fuse_model(model)

    model.train()
    model.qconfig = get_qat_qconfig_int8_int8()
    tq.prepare_qat(model, inplace=True)

    model.eval()
    return model


# ============================================================
# Dataset transform
# ============================================================
def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])


# ============================================================
# Binary / alignment helpers
# ============================================================
def align_up(x: int, align: int = ALIGN_BYTES) -> int:
    return ((x + align - 1) // align) * align


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def crc32_u32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def write_with_header(
    path: str,
    magic8: bytes,
    payload: bytes,
    version: int = FORMAT_VERSION,
    extra0: int = 0,
    extra1: int = 0,
    extra2: int = 0,
    extra3: int = 0,
):
    if len(magic8) != 8:
        raise ValueError(f"magic must be exactly 8 bytes, got {len(magic8)}")

    payload_crc = crc32_u32(payload)
    total_bytes = HEADER_BYTES + len(payload)

    header = bytearray(HEADER_BYTES)
    struct.pack_into(
        ENDIAN + "8sIIQQIIIIII8s",
        header,
        0,
        magic8,
        version,
        HEADER_BYTES,
        len(payload),
        total_bytes,
        payload_crc,
        0,
        extra0,
        extra1,
        extra2,
        extra3,
        b"\x00" * 8,
    )

    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)


# ============================================================
# QParam / tensor helpers
# ============================================================
def tensor_dtype_code_from_dtype(dtype: torch.dtype) -> int:
    if dtype == torch.int8:
        return DTYPE_INT8
    if dtype == torch.uint8:
        return DTYPE_UINT8
    if dtype == torch.int32:
        return DTYPE_INT32
    if dtype == torch.float32:
        return DTYPE_FLOAT32
    raise TypeError(f"Unsupported dtype for export: {dtype}")


def qscheme_code_from_obj(qscheme) -> int:
    if qscheme is None:
        return QSCHEME_NONE
    if qscheme == torch.per_tensor_affine:
        return QSCHEME_PER_TENSOR_AFFINE
    if qscheme == torch.per_channel_affine:
        return QSCHEME_PER_CHANNEL_AFFINE
    if qscheme == torch.per_channel_affine_float_qparams:
        return QSCHEME_PER_CHANNEL_AFFINE
    if qscheme == torch.per_tensor_symmetric:
        return QSCHEME_PER_TENSOR_SYMMETRIC
    if qscheme == torch.per_channel_symmetric:
        return QSCHEME_PER_CHANNEL_SYMMETRIC
    raise TypeError(f"Unsupported qscheme: {qscheme}")


def as_int_list(x: Any) -> List[int]:
    if isinstance(x, int):
        return [x]
    return list(x)


def get_tensor_shape4(t: Optional[torch.Tensor]) -> Tuple[int, int, int, int]:
    if t is None:
        return (0, 0, 0, 0)
    shape = list(t.shape)
    if len(shape) == 0:
        return (1, 1, 1, 1)
    if len(shape) == 1:
        return (1, shape[0], 1, 1)
    if len(shape) == 2:
        return (1, shape[0], shape[1], 1)
    if len(shape) == 3:
        return (shape[0], shape[1], shape[2], 1)
    if len(shape) >= 4:
        return (shape[0], shape[1], shape[2], shape[3])
    raise RuntimeError("Unexpected shape")


def float_tensor_to_bytes(t: torch.Tensor) -> bytes:
    arr = t.detach().contiguous().cpu().numpy().astype(np.float32, copy=False)
    return arr.tobytes(order="C")


def int32_tensor_to_bytes(t: torch.Tensor) -> bytes:
    arr = t.detach().contiguous().cpu().numpy().astype(np.int32, copy=False)
    return arr.tobytes(order="C")


def int8_tensor_to_bytes(t: torch.Tensor) -> bytes:
    arr = t.detach().contiguous().cpu().numpy().astype(np.int8, copy=False)
    return arr.tobytes(order="C")


def np_float32_to_bytes(arr: np.ndarray) -> bytes:
    arr = np.asarray(arr, dtype=np.float32)
    return arr.tobytes(order="C")


def np_int32_to_bytes(arr: np.ndarray) -> bytes:
    arr = np.asarray(arr, dtype=np.int32)
    return arr.tobytes(order="C")


def np_int8_to_bytes(arr: np.ndarray) -> bytes:
    arr = np.asarray(arr, dtype=np.int8)
    return arr.tobytes(order="C")


def clamp_round_tensor(x: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    return torch.clamp(torch.round(x), qmin, qmax)


def reshape_qparams_for_broadcast(
    vec: torch.Tensor,
    axis: int,
    ndim: int,
) -> torch.Tensor:
    shape = [1] * ndim
    shape[axis] = vec.numel()
    return vec.reshape(shape)


def quantize_tensor_with_qparams(
    x_fp32: torch.Tensor,
    qparams: Dict[str, Any],
) -> torch.Tensor:
    """
    输出整数张量（int8 或 int32 这里只用 int8）。
    """
    if not qparams["is_quantized"]:
        raise RuntimeError("qparams says tensor is not quantized")

    qmin = int(qparams["quant_min"])
    qmax = int(qparams["quant_max"])
    dtype = qparams["dtype"]

    if dtype != torch.qint8:
        raise RuntimeError(f"Current exporter expects qint8, got {dtype}")

    x = x_fp32.detach().cpu().to(torch.float32)

    if qparams["qscheme_code"] in (QSCHEME_PER_TENSOR_AFFINE, QSCHEME_PER_TENSOR_SYMMETRIC):
        scale = float(qparams["scale"])
        zp = int(qparams["zero_point"])
        q = clamp_round_tensor(x / scale + zp, qmin, qmax).to(torch.int32)
        return q.to(torch.int8)

    if qparams["qscheme_code"] in (QSCHEME_PER_CHANNEL_AFFINE, QSCHEME_PER_CHANNEL_SYMMETRIC):
        scales = torch.as_tensor(qparams["scales"], dtype=torch.float32)
        zps = torch.as_tensor(qparams["zero_points"], dtype=torch.float32)
        axis = int(qparams["axis"])

        scales_bc = reshape_qparams_for_broadcast(scales, axis, x.ndim)
        zps_bc = reshape_qparams_for_broadcast(zps, axis, x.ndim)
        q = clamp_round_tensor(x / scales_bc + zps_bc, qmin, qmax).to(torch.int32)
        return q.to(torch.int8)

    raise RuntimeError(f"Unsupported qscheme_code: {qparams['qscheme_code']}")


def get_fake_quant_module_qparams(fake_quant_module: Optional[nn.Module]) -> Dict[str, Any]:
    info = {
        "is_quantized": False,
        "qscheme_code": QSCHEME_NONE,
        "dtype": None,
        "quant_min": None,
        "quant_max": None,
        "scale": None,
        "zero_point": None,
        "scales": None,
        "zero_points": None,
        "axis": -1,
    }

    if fake_quant_module is None:
        return info

    # FakeQuantize 自身就带 dtype / qscheme / quant_min / quant_max
    if not hasattr(fake_quant_module, "dtype"):
        return info

    dtype = fake_quant_module.dtype
    qscheme = getattr(fake_quant_module, "qscheme", None)
    quant_min = int(getattr(fake_quant_module, "quant_min"))
    quant_max = int(getattr(fake_quant_module, "quant_max"))

    # 优先从缓冲区读取，避免 calculate_qparams 警告
    scale_buf = getattr(fake_quant_module, "scale", None)
    zp_buf = getattr(fake_quant_module, "zero_point", None)

    if scale_buf is None or zp_buf is None or scale_buf.numel() == 0 or zp_buf.numel() == 0:
        scale_buf, zp_buf = fake_quant_module.calculate_qparams()

    info["is_quantized"] = True
    info["qscheme_code"] = qscheme_code_from_obj(qscheme)
    info["dtype"] = dtype
    info["quant_min"] = quant_min
    info["quant_max"] = quant_max

    if info["qscheme_code"] in (QSCHEME_PER_TENSOR_AFFINE, QSCHEME_PER_TENSOR_SYMMETRIC):
        info["scale"] = float(scale_buf.detach().cpu().reshape(-1)[0].item())
        info["zero_point"] = int(zp_buf.detach().cpu().reshape(-1)[0].item())
        info["axis"] = -1
    elif info["qscheme_code"] in (QSCHEME_PER_CHANNEL_AFFINE, QSCHEME_PER_CHANNEL_SYMMETRIC):
        info["scales"] = scale_buf.detach().cpu().numpy().astype(np.float32)
        info["zero_points"] = zp_buf.detach().cpu().numpy().astype(np.int32)
        info["axis"] = int(getattr(fake_quant_module, "ch_axis", 0))
    else:
        raise RuntimeError(f"Unsupported qscheme in fake quant module: {qscheme}")

    return info


def get_module_output_qparams(module: Optional[nn.Module]) -> Dict[str, Any]:
    if module is None:
        return {
            "is_quantized": False,
            "qscheme_code": QSCHEME_NONE,
            "dtype": None,
            "quant_min": None,
            "quant_max": None,
            "scale": None,
            "zero_point": None,
            "scales": None,
            "zero_points": None,
            "axis": -1,
        }

    fake_quant = getattr(module, "activation_post_process", None)
    return get_fake_quant_module_qparams(fake_quant)


def get_module_weight_qparams(module: nn.Module) -> Dict[str, Any]:
    fake_quant = getattr(module, "weight_fake_quant", None)
    return get_fake_quant_module_qparams(fake_quant)


def get_module_weight_bias(module: nn.Module) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    weight = None
    bias = None

    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
        weight = module.weight

    if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
        bias = module.bias

    return weight, bias


def quantize_bias_to_int32(
    bias_fp32: Optional[torch.Tensor],
    input_q: Dict[str, Any],
    weight_q: Dict[str, Any],
    out_channels: int,
) -> Optional[torch.Tensor]:
    if bias_fp32 is None:
        return None

    if not input_q["is_quantized"]:
        raise RuntimeError("Input activation is not quantized; cannot derive int32 bias scale")

    if input_q["qscheme_code"] not in (QSCHEME_PER_TENSOR_AFFINE, QSCHEME_PER_TENSOR_SYMMETRIC):
        raise RuntimeError("Current exporter expects activation quantization to be per-tensor")

    bias_fp32 = bias_fp32.detach().cpu().to(torch.float32)
    input_scale = float(input_q["scale"])

    if weight_q["qscheme_code"] in (QSCHEME_PER_TENSOR_AFFINE, QSCHEME_PER_TENSOR_SYMMETRIC):
        ws = float(weight_q["scale"])
        denom = input_scale * ws
        bias_int32 = torch.round(bias_fp32 / denom).to(torch.int32)
        return bias_int32

    if weight_q["qscheme_code"] in (QSCHEME_PER_CHANNEL_AFFINE, QSCHEME_PER_CHANNEL_SYMMETRIC):
        w_scales = np.asarray(weight_q["scales"], dtype=np.float32)
        if len(w_scales) != out_channels:
            raise RuntimeError(
                f"Per-channel weight scales len mismatch: {len(w_scales)} vs out_channels {out_channels}"
            )
        denom = torch.from_numpy((input_scale * w_scales).astype(np.float32))
        bias_int32 = torch.round(bias_fp32 / denom).to(torch.int32)
        return bias_int32

    raise RuntimeError("Unsupported weight qscheme for bias quantization")


# ============================================================
# Layer descriptors
# ============================================================
@dataclass
class LayerDesc:
    layer_id: int
    name: str
    op_type: int
    op_name: str
    in_n: int
    in_c: int
    in_h: int
    in_w: int
    out_n: int
    out_c: int
    out_h: int
    out_w: int
    k_h: int
    k_w: int
    s_h: int
    s_w: int
    p_h: int
    p_w: int
    has_relu: int
    has_bias: int
    input_dtype: int
    output_dtype: int
    weight_offset: int
    weight_bytes: int
    bias_offset: int
    bias_bytes: int
    quant_record_index: int
    reserved0: int = 0
    reserved1: int = 0


@dataclass
class QuantDesc:
    layer_id: int
    input_qscheme: int
    weight_qscheme: int
    output_qscheme: int
    input_scale: float
    input_zero_point: int
    output_scale: float
    output_zero_point: int
    weight_scale_offset: int
    weight_scale_count: int
    weight_zp_offset: int
    weight_zp_count: int
    weight_scale_single: float
    weight_zero_point_single: int
    weight_axis: int
    reserved0: int = 0
    reserved1: int = 0


@dataclass
class TileDesc:
    layer_id: int
    op_type: int
    tile_id: int
    group_id: int
    array_dim: int
    flags: int
    oh_start: int
    ow_start: int
    oh_len: int
    ow_len: int
    oc_start: int
    oc_len: int
    k_start: int
    k_len: int
    m_dim: int
    n_dim: int
    k_dim: int
    input_bytes: int
    weight_bytes: int
    output_bytes: int
    reserved0: int = 0
    reserved1: int = 0


# ============================================================
# Exporter
# ============================================================
class HardwarePackageExporter:
    def __init__(self):
        self.layer_descs: List[LayerDesc] = []
        self.quant_descs: List[QuantDesc] = []
        self.tile_descs: List[TileDesc] = []

        self.manifest: Dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "endian": "little",
            "align_bytes": ALIGN_BYTES,
            "cfg_record_bytes": CFG_RECORD_BYTES,
            "quant_record_bytes": QUANT_RECORD_BYTES,
            "tile_record_bytes": TILE_RECORD_BYTES,
            "files": {},
            "layers": [],
            "sample": {},
            "notes": [
                "This exporter uses QAT fake-quant checkpoint, not torch.ao quantized backend.",
                "input.bin stores signed int8 activation after QuantStub fake-quant semantics.",
                "weight.bin stores signed int8 weights manually quantized from float weights + weight_fake_quant qparams.",
                "bias.bin stores int32 bias computed from bias_fp32 / (input_scale * weight_scale).",
                "golden_output.bin stores final float32 logits from fake-quant model dequant output.",
            ],
        }

        self.weight_blob = bytearray()
        self.bias_blob = bytearray()
        self.weight_scale_blob = bytearray()
        self.weight_zp_blob = bytearray()

        self.intermediate_files: List[str] = []

    def _append_aligned(self, blob: bytearray, payload: bytes, align: int = ALIGN_BYTES) -> Tuple[int, int]:
        offset = align_up(len(blob), align)
        if offset > len(blob):
            blob.extend(b"\x00" * (offset - len(blob)))
        blob.extend(payload)
        return offset, len(payload)

    def _pack_layer_record(self, rec: LayerDesc) -> bytes:
        values = [
            rec.layer_id,
            rec.op_type,
            rec.in_n, rec.in_c, rec.in_h, rec.in_w,
            rec.out_n, rec.out_c, rec.out_h, rec.out_w,
            rec.k_h, rec.k_w,
            rec.s_h, rec.s_w,
            rec.p_h, rec.p_w,
            rec.has_relu,
            rec.has_bias,
            rec.input_dtype,
            rec.output_dtype,
            rec.weight_offset,
            rec.weight_bytes,
            rec.bias_offset,
            rec.bias_bytes,
            rec.quant_record_index,
            rec.reserved0,
            rec.reserved1,
            0, 0, 0, 0, 0,
        ]
        assert len(values) == 32
        return struct.pack(ENDIAN + "32I", *values)

    def _pack_quant_record(self, rec: QuantDesc) -> bytes:
        payload = bytearray(QUANT_RECORD_BYTES)
        struct.pack_into(
            ENDIAN + "IIIIfIfIIIIIfiiII",
            payload,
            0,
            rec.layer_id,
            rec.input_qscheme,
            rec.weight_qscheme,
            rec.output_qscheme,
            rec.input_scale,
            rec.input_zero_point,
            rec.output_scale,
            rec.output_zero_point,
            rec.weight_scale_offset,
            rec.weight_scale_count,
            rec.weight_zp_offset,
            rec.weight_zp_count,
            rec.weight_scale_single,
            rec.weight_zero_point_single,
            rec.weight_axis,
            rec.reserved0,
            rec.reserved1,
        )
        return bytes(payload)

    def _write_cfg_files(self, export_dir: str):
        layer_payload = b"".join(self._pack_layer_record(r) for r in self.layer_descs)
        quant_payload = b"".join(self._pack_quant_record(r) for r in self.quant_descs)

        write_with_header(
            os.path.join(export_dir, "layer_cfg.bin"),
            MAGIC_LAYER_CFG,
            layer_payload,
            extra0=len(self.layer_descs),
            extra1=CFG_RECORD_BYTES,
            extra2=ALIGN_BYTES,
            extra3=crc32_u32(layer_payload),
        )
        write_with_header(
            os.path.join(export_dir, "quant_cfg.bin"),
            MAGIC_QUANT_CFG,
            quant_payload,
            extra0=len(self.quant_descs),
            extra1=QUANT_RECORD_BYTES,
            extra2=ALIGN_BYTES,
            extra3=crc32_u32(quant_payload),
        )

        self.manifest["files"]["layer_cfg.bin"] = {
            "type": "layer_cfg",
            "records": len(self.layer_descs),
            "record_bytes": CFG_RECORD_BYTES,
        }
        self.manifest["files"]["quant_cfg.bin"] = {
            "type": "quant_cfg",
            "records": len(self.quant_descs),
            "record_bytes": QUANT_RECORD_BYTES,
        }

    def _write_param_files(self, export_dir: str):
        write_with_header(
            os.path.join(export_dir, "weight.bin"),
            MAGIC_WEIGHT,
            bytes(self.weight_blob),
            extra0=len(self.weight_blob),
            extra1=ALIGN_BYTES,
            extra2=crc32_u32(bytes(self.weight_blob)),
            extra3=0,
        )
        write_with_header(
            os.path.join(export_dir, "bias.bin"),
            MAGIC_BIAS,
            bytes(self.bias_blob),
            extra0=len(self.bias_blob),
            extra1=ALIGN_BYTES,
            extra2=crc32_u32(bytes(self.bias_blob)),
            extra3=0,
        )

        self.manifest["files"]["weight.bin"] = {
            "type": "weight_int8",
            "payload_bytes": len(self.weight_blob),
            "align_bytes": ALIGN_BYTES,
        }
        self.manifest["files"]["bias.bin"] = {
            "type": "bias_int32",
            "payload_bytes": len(self.bias_blob),
            "align_bytes": ALIGN_BYTES,
        }

    def _write_quant_aux_files(self, export_dir: str):
        write_with_header(
            os.path.join(export_dir, "weight_scales.bin"),
            b"WSCAL001",
            bytes(self.weight_scale_blob),
            extra0=len(self.weight_scale_blob),
            extra1=ALIGN_BYTES,
            extra2=crc32_u32(bytes(self.weight_scale_blob)),
            extra3=0,
        )
        write_with_header(
            os.path.join(export_dir, "weight_zero_points.bin"),
            b"WZPNT001",
            bytes(self.weight_zp_blob),
            extra0=len(self.weight_zp_blob),
            extra1=ALIGN_BYTES,
            extra2=crc32_u32(bytes(self.weight_zp_blob)),
            extra3=0,
        )

        self.manifest["files"]["weight_scales.bin"] = {
            "type": "weight_scales_float32",
            "payload_bytes": len(self.weight_scale_blob),
            "align_bytes": ALIGN_BYTES,
        }
        self.manifest["files"]["weight_zero_points.bin"] = {
            "type": "weight_zero_points_int32",
            "payload_bytes": len(self.weight_zp_blob),
            "align_bytes": ALIGN_BYTES,
        }

    def _write_input_and_output(
        self,
        export_dir: str,
        input_int8: torch.Tensor,
        input_qparams: Dict[str, Any],
        logits_fp32: torch.Tensor,
    ):
        input_payload = int8_tensor_to_bytes(input_int8)
        write_with_header(
            os.path.join(export_dir, "input.bin"),
            MAGIC_INPUT,
            input_payload,
            extra0=int(input_int8.numel()),
            extra1=DTYPE_INT8,
            extra2=crc32_u32(input_payload),
            extra3=0,
        )

        output_payload = float_tensor_to_bytes(logits_fp32)
        write_with_header(
            os.path.join(export_dir, "golden_output.bin"),
            MAGIC_GOLDEN,
            output_payload,
            extra0=int(logits_fp32.numel()),
            extra1=DTYPE_FLOAT32,
            extra2=crc32_u32(output_payload),
            extra3=0,
        )

        self.manifest["files"]["input.bin"] = {
            "type": "input_quantized_activation",
            "shape": list(input_int8.shape),
            "dtype": "int8",
            "scale": input_qparams["scale"],
            "zero_point": input_qparams["zero_point"],
            "qscheme_code": input_qparams["qscheme_code"],
        }
        self.manifest["files"]["golden_output.bin"] = {
            "type": "final_logits_fp32",
            "shape": list(logits_fp32.shape),
            "dtype": "float32",
        }

    def _write_one_intermediate_int8(self, export_dir: str, layer_id: int, out_int8: torch.Tensor):
        fname = f"layer_{layer_id:03d}_out.bin"
        path = os.path.join(export_dir, fname)

        payload = int8_tensor_to_bytes(out_int8)
        write_with_header(
            path,
            MAGIC_LAYER_OUT,
            payload,
            extra0=int(out_int8.numel()),
            extra1=DTYPE_INT8,
            extra2=crc32_u32(payload),
            extra3=layer_id,
        )

        self.intermediate_files.append(fname)
        self.manifest["files"][fname] = {
            "type": "layer_output",
            "layer_id": layer_id,
            "shape": list(out_int8.shape),
            "is_quantized": True,
            "dtype": "int8",
        }

    def _write_one_intermediate_float(self, export_dir: str, layer_id: int, out_fp32: torch.Tensor):
        fname = f"layer_{layer_id:03d}_out.bin"
        path = os.path.join(export_dir, fname)

        payload = float_tensor_to_bytes(out_fp32)
        write_with_header(
            path,
            MAGIC_LAYER_OUT,
            payload,
            extra0=int(out_fp32.numel()),
            extra1=DTYPE_FLOAT32,
            extra2=crc32_u32(payload),
            extra3=layer_id,
        )

        self.intermediate_files.append(fname)
        self.manifest["files"][fname] = {
            "type": "layer_output",
            "layer_id": layer_id,
            "shape": list(out_fp32.shape),
            "is_quantized": False,
            "dtype": "float32",
        }

    def _append_weight_qparams(self, weight_q: Dict[str, Any]) -> Tuple[int, int, int, int]:
        if weight_q["qscheme_code"] in (QSCHEME_PER_CHANNEL_AFFINE, QSCHEME_PER_CHANNEL_SYMMETRIC):
            scales = np.asarray(weight_q["scales"], dtype=np.float32)
            zps = np.asarray(weight_q["zero_points"], dtype=np.int32)
            s_off, _ = self._append_aligned(self.weight_scale_blob, np_float32_to_bytes(scales), ALIGN_BYTES)
            z_off, _ = self._append_aligned(self.weight_zp_blob, np_int32_to_bytes(zps), ALIGN_BYTES)
            return s_off, len(scales), z_off, len(zps)

        if weight_q["qscheme_code"] in (QSCHEME_PER_TENSOR_AFFINE, QSCHEME_PER_TENSOR_SYMMETRIC):
            return 0, 0, 0, 0

        return 0, 0, 0, 0

    def _pack_tile_record(self, rec: TileDesc) -> bytes:
        values = [
            rec.layer_id,
            rec.op_type,
            rec.tile_id,
            rec.group_id,
            rec.array_dim,
            rec.flags,
            rec.oh_start,
            rec.ow_start,
            rec.oh_len,
            rec.ow_len,
            rec.oc_start,
            rec.oc_len,
            rec.k_start,
            rec.k_len,
            rec.m_dim,
            rec.n_dim,
            rec.k_dim,
            rec.input_bytes,
            rec.weight_bytes,
            rec.output_bytes,
            rec.reserved0,
            rec.reserved1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
        assert len(values) == 32
        return struct.pack(ENDIAN + "32I", *values)

    def _write_tile_plan_file(self, export_dir: str):
        payload = b"".join(self._pack_tile_record(r) for r in self.tile_descs)
        write_with_header(
            os.path.join(export_dir, "tile_plan.bin"),
            MAGIC_TILE_PLAN,
            payload,
            extra0=len(self.tile_descs),
            extra1=TILE_RECORD_BYTES,
            extra2=ALIGN_BYTES,
            extra3=crc32_u32(payload),
        )
        self.manifest["files"]["tile_plan.bin"] = {
            "type": "tile_plan",
            "records": len(self.tile_descs),
            "record_bytes": TILE_RECORD_BYTES,
        }

    def _choose_array_dim(self, m_dim: int, n_dim: int) -> int:
        candidates = [4, 8, 16, 32]
        best = 4
        best_key = None
        for a in candidates:
            util_m = min(m_dim, a) / a
            util_n = min(n_dim, a) / a
            util = util_m * util_n
            effective = min(m_dim, a) * min(n_dim, a)
            key = (util, effective, a)
            if best_key is None or key > best_key:
                best_key = key
                best = a
        return best

    def _conv_candidate_ok(self, layer: LayerDesc, oh_len: int, ow_len: int, oc_len: int, cin_len: int) -> Tuple[bool, int, int, int]:
        kh = layer.k_h
        kw = layer.k_w
        sh = max(layer.s_h, 1)
        sw = max(layer.s_w, 1)
        patch_h = (oh_len - 1) * sh + kh
        patch_w = (ow_len - 1) * sw + kw
        input_bytes = cin_len * patch_h * patch_w
        weight_bytes = oc_len * cin_len * kh * kw
        output_bytes = oh_len * ow_len * oc_len * 4
        ok = (input_bytes <= 2048 and weight_bytes <= 2048 and output_bytes <= 2048)
        return ok, input_bytes, weight_bytes, output_bytes

    def _linear_candidate_ok(self, oc_len: int, k_len: int) -> Tuple[bool, int, int, int]:
        input_bytes = k_len
        weight_bytes = oc_len * k_len
        output_bytes = oc_len * 4
        ok = (input_bytes <= 2048 and weight_bytes <= 2048 and output_bytes <= 2048)
        return ok, input_bytes, weight_bytes, output_bytes

    def _pick_conv_tile_shape(self, layer: LayerDesc) -> Tuple[int, int, int, int, int, int]:
        best = None
        total_ic = layer.in_c
        khkw = layer.k_h * layer.k_w
        for oc_len in range(min(layer.out_c, 32), 0, -1):
            for oh_len in range(min(layer.out_h, 8), 0, -1):
                for ow_len in range(min(layer.out_w, 8), 0, -1):
                    max_cin = min(total_ic, 2048 // max(1, oc_len * khkw))
                    for cin_len in range(max_cin, 0, -1):
                        ok, ib, wb, ob = self._conv_candidate_ok(layer, oh_len, ow_len, oc_len, cin_len)
                        if not ok:
                            continue
                        m_dim = oh_len * ow_len
                        n_dim = oc_len
                        k_dim = cin_len * khkw
                        array_dim = self._choose_array_dim(m_dim, n_dim)
                        util = (min(m_dim, array_dim) * min(n_dim, array_dim)) / float(array_dim * array_dim)
                        key = (util, array_dim, n_dim, m_dim, k_dim, cin_len)
                        if best is None or key > best[0]:
                            best = (key, (oh_len, ow_len, oc_len, cin_len, ib, wb, ob))
        if best is None:
            raise RuntimeError(f"No legal conv tile found for layer_id={layer.layer_id}")
        return best[1]

    def _pick_linear_tile_shape(self, layer: LayerDesc) -> Tuple[int, int, int, int, int]:
        total_k = layer.in_c * max(layer.in_h, 1) * max(layer.in_w, 1)
        best = None
        for oc_len in range(min(layer.out_c, 32), 0, -1):
            max_k = min(total_k, 2048 // max(1, oc_len))
            for k_len in range(max_k, 0, -1):
                ok, ib, wb, ob = self._linear_candidate_ok(oc_len, k_len)
                if not ok:
                    continue
                m_dim = 1
                n_dim = oc_len
                array_dim = self._choose_array_dim(m_dim, n_dim)
                util = (min(m_dim, array_dim) * min(n_dim, array_dim)) / float(array_dim * array_dim)
                key = (util, array_dim, n_dim, k_len)
                if best is None or key > best[0]:
                    best = (key, (oc_len, k_len, ib, wb, ob))
        if best is None:
            raise RuntimeError(f"No legal linear tile found for layer_id={layer.layer_id}")
        return best[1]

    def _build_tile_plan(self):
        self.tile_descs = []
        next_tile_id = 0
        next_group_id = 0
        for layer in self.layer_descs:
            if layer.op_type == OP_CONV_RELU:
                oh_len_base, ow_len_base, oc_len_base, cin_len_base, _, _, _ = self._pick_conv_tile_shape(layer)
                khkw = layer.k_h * layer.k_w
                for oc_start in range(0, layer.out_c, oc_len_base):
                    oc_len = min(oc_len_base, layer.out_c - oc_start)
                    for oh_start in range(0, layer.out_h, oh_len_base):
                        oh_len = min(oh_len_base, layer.out_h - oh_start)
                        for ow_start in range(0, layer.out_w, ow_len_base):
                            ow_len = min(ow_len_base, layer.out_w - ow_start)
                            group_id = next_group_id
                            next_group_id += 1
                            cin_start = 0
                            while cin_start < layer.in_c:
                                cin_len = min(cin_len_base, layer.in_c - cin_start)
                                ok, ib, wb, ob = self._conv_candidate_ok(layer, oh_len, ow_len, oc_len, cin_len)
                                if not ok:
                                    raise RuntimeError(f"Illegal conv tile after selection at layer {layer.layer_id}")
                                m_dim = oh_len * ow_len
                                n_dim = oc_len
                                k_dim = cin_len * khkw
                                flags = 0
                                if cin_start > 0:
                                    flags |= 1
                                if cin_start + cin_len >= layer.in_c:
                                    flags |= 2
                                array_dim = self._choose_array_dim(m_dim, n_dim)
                                self.tile_descs.append(TileDesc(
                                    layer_id=layer.layer_id,
                                    op_type=layer.op_type,
                                    tile_id=next_tile_id,
                                    group_id=group_id,
                                    array_dim=array_dim,
                                    flags=flags,
                                    oh_start=oh_start,
                                    ow_start=ow_start,
                                    oh_len=oh_len,
                                    ow_len=ow_len,
                                    oc_start=oc_start,
                                    oc_len=oc_len,
                                    k_start=cin_start * khkw,
                                    k_len=cin_len * khkw,
                                    m_dim=m_dim,
                                    n_dim=n_dim,
                                    k_dim=k_dim,
                                    input_bytes=ib,
                                    weight_bytes=wb,
                                    output_bytes=ob,
                                ))
                                next_tile_id += 1
                                cin_start += cin_len
            elif layer.op_type == OP_LINEAR:
                oc_len_base, k_len_base, _, _, _ = self._pick_linear_tile_shape(layer)
                total_k = layer.in_c * max(layer.in_h, 1) * max(layer.in_w, 1)
                for oc_start in range(0, layer.out_c, oc_len_base):
                    oc_len = min(oc_len_base, layer.out_c - oc_start)
                    group_id = next_group_id
                    next_group_id += 1
                    k_start = 0
                    while k_start < total_k:
                        k_len = min(k_len_base, total_k - k_start)
                        ok, ib, wb, ob = self._linear_candidate_ok(oc_len, k_len)
                        if not ok:
                            raise RuntimeError(f"Illegal linear tile after selection at layer {layer.layer_id}")
                        m_dim = 1
                        n_dim = oc_len
                        flags = 0
                        if k_start > 0:
                            flags |= 1
                        if k_start + k_len >= total_k:
                            flags |= 2
                        array_dim = self._choose_array_dim(m_dim, n_dim)
                        self.tile_descs.append(TileDesc(
                            layer_id=layer.layer_id,
                            op_type=layer.op_type,
                            tile_id=next_tile_id,
                            group_id=group_id,
                            array_dim=array_dim,
                            flags=flags,
                            oh_start=0,
                            ow_start=0,
                            oh_len=1,
                            ow_len=1,
                            oc_start=oc_start,
                            oc_len=oc_len,
                            k_start=k_start,
                            k_len=k_len,
                            m_dim=m_dim,
                            n_dim=n_dim,
                            k_dim=k_len,
                            input_bytes=ib,
                            weight_bytes=wb,
                            output_bytes=ob,
                        ))
                        next_tile_id += 1
                        k_start += k_len

    def export(self):
        ensure_dir(EXPORT_DIR)
        torch.manual_seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        # --------------------------------------------------------
        # 1) Rebuild QAT fake-quant model and load checkpoint
        # --------------------------------------------------------
        model = build_qat_fakequant_model(width_mult=WIDTH_MULT)
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        if "model_state_dict" not in ckpt:
            raise RuntimeError("Checkpoint must contain key 'model_state_dict'")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # --------------------------------------------------------
        # 2) Get one CIFAR-10 sample
        # --------------------------------------------------------
        transform = get_test_transform()
        dataset_raw = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=None)
        dataset_tensor = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)

        sample_index = RANDOM_SEED % len(dataset_tensor)
        raw_img, true_label = dataset_raw[sample_index]
        input_fp32, _ = dataset_tensor[sample_index]
        input_fp32 = input_fp32.unsqueeze(0).cpu()

        layers_info_for_manifest = []

        # --------------------------------------------------------
        # 3) QuantStub output
        # --------------------------------------------------------
        current = model.quant(input_fp32)  # float tensor, but fake-quant semantics already applied
        quant0_q = get_module_output_qparams(model.quant)
        if not quant0_q["is_quantized"]:
            raise RuntimeError("QuantStub activation_post_process qparams not found")

        input_int8 = quantize_tensor_with_qparams(current, quant0_q)
        self._write_one_intermediate_int8(EXPORT_DIR, 0, input_int8)

        layer0 = LayerDesc(
            layer_id=0,
            name="quant",
            op_type=OP_QUANTIZE,
            op_name="quantize",
            in_n=1, in_c=3, in_h=32, in_w=32,
            out_n=current.shape[0], out_c=current.shape[1], out_h=current.shape[2], out_w=current.shape[3],
            k_h=0, k_w=0, s_h=0, s_w=0, p_h=0, p_w=0,
            has_relu=0,
            has_bias=0,
            input_dtype=DTYPE_FLOAT32,
            output_dtype=DTYPE_INT8,
            weight_offset=0,
            weight_bytes=0,
            bias_offset=0,
            bias_bytes=0,
            quant_record_index=0,
        )
        quant0 = QuantDesc(
            layer_id=0,
            input_qscheme=QSCHEME_NONE,
            weight_qscheme=QSCHEME_NONE,
            output_qscheme=quant0_q["qscheme_code"],
            input_scale=0.0,
            input_zero_point=0,
            output_scale=float(quant0_q["scale"]),
            output_zero_point=int(quant0_q["zero_point"]),
            weight_scale_offset=0,
            weight_scale_count=0,
            weight_zp_offset=0,
            weight_zp_count=0,
            weight_scale_single=0.0,
            weight_zero_point_single=0,
            weight_axis=-1,
        )
        self.layer_descs.append(layer0)
        self.quant_descs.append(quant0)
        layers_info_for_manifest.append({"layer_id": 0, "name": "quant", "op_name": "quantize"})

        next_layer_id = 1
        current_qparams = quant0_q

        def append_layer_common(
            layer_id: int,
            name: str,
            op_type: int,
            has_relu: int,
            input_tensor: torch.Tensor,
            output_tensor: torch.Tensor,
            input_q: Dict[str, Any],
            output_q: Dict[str, Any],
            weight_q: Dict[str, Any],
            weight_offset: int,
            weight_bytes: int,
            bias_offset: int,
            bias_bytes: int,
            explicit_kernel: Tuple[int, int] = (0, 0),
            explicit_stride: Tuple[int, int] = (0, 0),
            explicit_padding: Tuple[int, int] = (0, 0),
            module: Optional[nn.Module] = None,
        ):
            if op_type == OP_LINEAR:
                if input_tensor.ndim != 2 or output_tensor.ndim != 2:
                    raise RuntimeError(
                        f"Linear layer expects 2D tensors, got input={input_tensor.shape}, output={output_tensor.shape}"
                    )
                in_n = int(input_tensor.shape[0])
                in_c = int(input_tensor.shape[1])
                in_h = 1
                in_w = 1

                out_n = int(output_tensor.shape[0])
                out_c = int(output_tensor.shape[1])
                out_h = 1
                out_w = 1
            else:
                in_n, in_c, in_h, in_w = get_tensor_shape4(input_tensor)
                out_n, out_c, out_h, out_w = get_tensor_shape4(output_tensor)

            if explicit_kernel != (0, 0):
                k_h, k_w = explicit_kernel
                s_h, s_w = explicit_stride
                p_h, p_w = explicit_padding
            else:
                if module is not None and hasattr(module, "kernel_size"):
                    ks = as_int_list(module.kernel_size)
                    k_h = ks[0]
                    k_w = ks[1] if len(ks) > 1 else ks[0]
                else:
                    k_h = k_w = 0

                if module is not None and hasattr(module, "stride"):
                    st = as_int_list(module.stride)
                    s_h = st[0]
                    s_w = st[1] if len(st) > 1 else st[0]
                else:
                    s_h = s_w = 0

                if module is not None and hasattr(module, "padding"):
                    pd = as_int_list(module.padding)
                    p_h = pd[0]
                    p_w = pd[1] if len(pd) > 1 else pd[0]
                else:
                    p_h = p_w = 0

            scale_off, scale_cnt, zp_off, zp_cnt = self._append_weight_qparams(weight_q)

            if weight_q["qscheme_code"] in (QSCHEME_PER_TENSOR_AFFINE, QSCHEME_PER_TENSOR_SYMMETRIC):
                weight_scale_single = float(weight_q["scale"])
                weight_zero_point_single = int(weight_q["zero_point"])
                weight_axis = -1
            elif weight_q["qscheme_code"] in (QSCHEME_PER_CHANNEL_AFFINE, QSCHEME_PER_CHANNEL_SYMMETRIC):
                weight_scale_single = 0.0
                weight_zero_point_single = 0
                weight_axis = int(weight_q["axis"])
            else:
                weight_scale_single = 0.0
                weight_zero_point_single = 0
                weight_axis = -1

            layer_desc = LayerDesc(
                layer_id=layer_id,
                name=name,
                op_type=op_type,
                op_name=name,
                in_n=in_n, in_c=in_c, in_h=in_h, in_w=in_w,
                out_n=out_n, out_c=out_c, out_h=out_h, out_w=out_w,
                k_h=k_h, k_w=k_w,
                s_h=s_h, s_w=s_w,
                p_h=p_h, p_w=p_w,
                has_relu=has_relu,
                has_bias=1 if bias_bytes > 0 else 0,
                input_dtype=DTYPE_INT8 if input_q["is_quantized"] else DTYPE_FLOAT32,
                output_dtype=DTYPE_INT8 if output_q["is_quantized"] else DTYPE_FLOAT32,
                weight_offset=weight_offset,
                weight_bytes=weight_bytes,
                bias_offset=bias_offset,
                bias_bytes=bias_bytes,
                quant_record_index=len(self.quant_descs),
            )
            quant_desc = QuantDesc(
                layer_id=layer_id,
                input_qscheme=input_q["qscheme_code"],
                weight_qscheme=weight_q["qscheme_code"],
                output_qscheme=output_q["qscheme_code"],
                input_scale=float(input_q["scale"]) if input_q["scale"] is not None else 0.0,
                input_zero_point=int(input_q["zero_point"]) if input_q["zero_point"] is not None else 0,
                output_scale=float(output_q["scale"]) if output_q["scale"] is not None else 0.0,
                output_zero_point=int(output_q["zero_point"]) if output_q["zero_point"] is not None else 0,
                weight_scale_offset=scale_off,
                weight_scale_count=scale_cnt,
                weight_zp_offset=zp_off,
                weight_zp_count=zp_cnt,
                weight_scale_single=weight_scale_single,
                weight_zero_point_single=weight_zero_point_single,
                weight_axis=weight_axis,
            )

            self.layer_descs.append(layer_desc)
            self.quant_descs.append(quant_desc)
            layers_info_for_manifest.append({
                "layer_id": layer_id,
                "name": name,
                "op_name": name,
                "weight_offset": weight_offset,
                "weight_bytes": weight_bytes,
                "bias_offset": bias_offset,
                "bias_bytes": bias_bytes,
                "quant_record_index": layer_desc.quant_record_index,
            })

        def handle_quantized_weight_module(
            module: nn.Module,
            name: str,
            op_type: int,
            has_relu: int,
        ):
            nonlocal current, current_qparams, next_layer_id

            input_tensor = current
            input_q = current_qparams

            # 1) 仍然保留 fake-quant float 前向，只用于驱动后续层
            output_tensor = module(current)

            output_q = get_module_output_qparams(module)
            if not output_q["is_quantized"]:
                raise RuntimeError(f"Layer {name} missing activation_post_process qparams")

            # 2) 读取并手动量化权重
            weight_fp32, bias_fp32 = get_module_weight_bias(module)
            if weight_fp32 is None:
                raise RuntimeError(f"Layer {name} has no weight tensor")

            weight_q = get_module_weight_qparams(module)
            if not weight_q["is_quantized"]:
                raise RuntimeError(f"Layer {name} missing weight_fake_quant qparams")

            input_int8_t = quantize_tensor_with_qparams(input_tensor, input_q)
            weight_int8_t = quantize_tensor_with_qparams(weight_fp32, weight_q)

            input_int8 = input_int8_t.cpu().numpy().astype(np.int8)
            weight_int8 = weight_int8_t.cpu().numpy().astype(np.int8)

            # 3) bias int32
            out_channels = int(weight_fp32.shape[0])
            bias_int32_t = quantize_bias_to_int32(bias_fp32, input_q, weight_q, out_channels)
            if bias_int32_t is None:
                raise RuntimeError(f"Layer {name} expects bias in current exporter")
            bias_int32 = bias_int32_t.cpu().numpy().astype(np.int32)

            # 4) 导出权重和 bias
            weight_offset, weight_bytes = self._append_aligned(
                self.weight_blob,
                int8_tensor_to_bytes(weight_int8_t),
                ALIGN_BYTES,
            )

            bias_offset, bias_bytes = self._append_aligned(
                self.bias_blob,
                int32_tensor_to_bytes(bias_int32_t),
                ALIGN_BYTES,
            )

            # 5) 用整数链路生成 layer_xxx_out.bin
            if weight_q["qscheme_code"] not in (QSCHEME_PER_CHANNEL_AFFINE, QSCHEME_PER_CHANNEL_SYMMETRIC):
                raise RuntimeError(f"Current exporter expects per-channel weight quant for weighted layers, got {weight_q['qscheme_code']}")

            weight_scales = np.asarray(weight_q["scales"], dtype=np.float32)
            weight_zps = np.asarray(weight_q["zero_points"], dtype=np.int32)

            if op_type == OP_CONV_RELU:
                ks = as_int_list(module.kernel_size)
                st = as_int_list(module.stride)
                pd = as_int_list(module.padding)

                k_h = ks[0]
                k_w = ks[1] if len(ks) > 1 else ks[0]
                s_h = st[0]
                s_w = st[1] if len(st) > 1 else st[0]
                p_h = pd[0]
                p_w = pd[1] if len(pd) > 1 else pd[0]

                acc = conv2d_integer_im2col(
                    input_int8,
                    weight_int8,
                    bias_int32,
                    input_zero_point=int(input_q["zero_point"]),
                    weight_zero_points=weight_zps,
                    stride_h=s_h,
                    stride_w=s_w,
                    pad_h=p_h,
                    pad_w=p_w,
                )

                output_int8_np = requantize_per_channel(
                    acc,
                    input_scale=float(input_q["scale"]),
                    weight_scales=weight_scales,
                    output_scale=float(output_q["scale"]),
                    output_zero_point=int(output_q["zero_point"]),
                    out_dtype_code=DTYPE_INT8,
                )

                if has_relu:
                    output_int8_np = relu_quantized_np(
                        output_int8_np,
                        int(output_q["zero_point"]),
                        DTYPE_INT8,
                    )

            elif op_type == OP_LINEAR:
                acc = linear_integer(
                    input_int8,
                    weight_int8,
                    bias_int32,
                    input_zero_point=int(input_q["zero_point"]),
                    weight_zero_points=weight_zps,
                )

                output_int8_np = requantize_per_channel(
                    acc,
                    input_scale=float(input_q["scale"]),
                    weight_scales=weight_scales,
                    output_scale=float(output_q["scale"]),
                    output_zero_point=int(output_q["zero_point"]),
                    out_dtype_code=DTYPE_INT8,
                )

            else:
                raise RuntimeError(f"Unsupported weighted op_type={op_type} in exporter")

            output_int8_t = torch.from_numpy(output_int8_np.astype(np.int8))
            self._write_one_intermediate_int8(EXPORT_DIR, next_layer_id, output_int8_t)

            append_layer_common(
                layer_id=next_layer_id,
                name=name,
                op_type=op_type,
                has_relu=has_relu,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                input_q=input_q,
                output_q=output_q,
                weight_q=weight_q,
                weight_offset=weight_offset,
                weight_bytes=weight_bytes,
                bias_offset=bias_offset,
                bias_bytes=bias_bytes,
                module=module,
            )

            # 关键改动：
            # 后续层不再吃 fake-quant 原始 output_tensor，
            # 而是吃“output_int8 -> dequant”恢复出来的 float 张量
            current = dequantize_int_tensor_with_qparams(output_int8_np, output_q).to(torch.float32)
            current_qparams = output_q
            next_layer_id += 1

        def handle_passthrough_activation_module(
            module: nn.Module,
            name: str,
            op_type: int,
            has_relu: int,
        ):
            nonlocal current, current_qparams, next_layer_id

            input_tensor = current
            input_q = current_qparams
            output_tensor = module(current)

            output_q = input_q
            output_int8 = quantize_tensor_with_qparams(output_tensor, output_q)
            self._write_one_intermediate_int8(EXPORT_DIR, next_layer_id, output_int8)

            append_layer_common(
                layer_id=next_layer_id,
                name=name,
                op_type=op_type,
                has_relu=has_relu,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                input_q=input_q,
                output_q=output_q,
                weight_q={"qscheme_code": QSCHEME_NONE, "scale": None, "zero_point": None},
                weight_offset=0,
                weight_bytes=0,
                bias_offset=0,
                bias_bytes=0,
                module=module,
            )

            # 关键改动：后续层吃的是导出的 int8 对应的 dequant float
            current = dequantize_int_tensor_with_qparams(output_int8.cpu().numpy(), output_q).to(torch.float32)
            current_qparams = output_q
            next_layer_id += 1

        # --------------------------------------------------------
        # 4) stage1..stage4
        # --------------------------------------------------------
        for stage_name in ["stage1", "stage2", "stage3", "stage4"]:
            stage = getattr(model.model, stage_name)
            for idx, subm in enumerate(stage):
                sub_name = f"{stage_name}.{idx}"

                if hasattr(subm, "conv"):
                    handle_quantized_weight_module(subm.conv, sub_name, OP_CONV_RELU, has_relu=1)
                elif isinstance(subm, nn.MaxPool2d):
                    handle_passthrough_activation_module(subm, sub_name, OP_MAXPOOL, has_relu=0)
                else:
                    raise TypeError(f"Unsupported stage module in exporter: {sub_name} -> {type(subm)}")

        # --------------------------------------------------------
        # 5) avgpool
        # --------------------------------------------------------
        avg_in = current
        avg_out = model.model.avgpool(current)

        in_n, in_c, in_h, in_w = get_tensor_shape4(avg_in)
        out_n, out_c, out_h, out_w = get_tensor_shape4(avg_out)

        avg_q = current_qparams

        # 关键改动：
        # 不再用 float avg_out 直接 round 成 int8，
        # 而是和 replay 一样，在整数域做平均
        avg_in_int8 = quantize_tensor_with_qparams(avg_in, avg_q).cpu().numpy().astype(np.int8)

        k_h = in_h
        k_w = in_w
        s_h = in_h
        s_w = in_w

        avg_out_int8_np = avgpool_integer_same_qparams(
            avg_in_int8,
            k_h,
            k_w,
            s_h,
            s_w,
            DTYPE_INT8,
        )
        avg_out_int8 = torch.from_numpy(avg_out_int8_np.astype(np.int8))
        self._write_one_intermediate_int8(EXPORT_DIR, next_layer_id, avg_out_int8)

        self.layer_descs.append(LayerDesc(
            layer_id=next_layer_id,
            name="avgpool",
            op_type=OP_AVGPOOL,
            op_name="avgpool",
            in_n=in_n, in_c=in_c, in_h=in_h, in_w=in_w,
            out_n=out_n, out_c=out_c, out_h=out_h, out_w=out_w,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
            p_h=0,
            p_w=0,
            has_relu=0,
            has_bias=0,
            input_dtype=DTYPE_INT8,
            output_dtype=DTYPE_INT8,
            weight_offset=0,
            weight_bytes=0,
            bias_offset=0,
            bias_bytes=0,
            quant_record_index=len(self.quant_descs),
        ))

        self.quant_descs.append(QuantDesc(
            layer_id=next_layer_id,
            input_qscheme=avg_q["qscheme_code"],
            weight_qscheme=QSCHEME_NONE,
            output_qscheme=avg_q["qscheme_code"],
            input_scale=float(avg_q["scale"]) if avg_q["scale"] is not None else 0.0,
            input_zero_point=int(avg_q["zero_point"]) if avg_q["zero_point"] is not None else 0,
            output_scale=float(avg_q["scale"]) if avg_q["scale"] is not None else 0.0,
            output_zero_point=int(avg_q["zero_point"]) if avg_q["zero_point"] is not None else 0,
            weight_scale_offset=0,
            weight_scale_count=0,
            weight_zp_offset=0,
            weight_zp_count=0,
            weight_scale_single=0.0,
            weight_zero_point_single=0,
            weight_axis=-1,
        ))

        layers_info_for_manifest.append({
            "layer_id": next_layer_id,
            "name": "avgpool",
            "op_name": "avgpool",
        })

        # 后续层继续吃“真正导出的 int8”对应的反量化结果
        current = dequantize_int_tensor_with_qparams(avg_out_int8_np, avg_q).to(torch.float32)
        current_qparams = avg_q
        next_layer_id += 1

        # --------------------------------------------------------
        # 6) flatten
        # --------------------------------------------------------
        flat_in = current
        flat_out = torch.flatten(current, 1)
        flat_q = current_qparams
        flat_int8 = quantize_tensor_with_qparams(flat_out, flat_q)
        self._write_one_intermediate_int8(EXPORT_DIR, next_layer_id, flat_int8)

        append_layer_common(
            layer_id=next_layer_id,
            name="flatten",
            op_type=OP_FLATTEN,
            has_relu=0,
            input_tensor=flat_in,
            output_tensor=flat_out,
            input_q=current_qparams,
            output_q=flat_q,
            weight_q={"qscheme_code": QSCHEME_NONE, "scale": None, "zero_point": None},
            weight_offset=0,
            weight_bytes=0,
            bias_offset=0,
            bias_bytes=0,
            module=None,
        )

        current = dequantize_int_tensor_with_qparams(flat_int8.cpu().numpy(), flat_q).to(torch.float32)
        current_qparams = flat_q
        next_layer_id += 1

        # --------------------------------------------------------
        # 7) classifier
        # --------------------------------------------------------
        handle_quantized_weight_module(model.model.classifier, "classifier", OP_LINEAR, has_relu=0)

        # --------------------------------------------------------
        # 8) dequant
        # --------------------------------------------------------
        deq_in = current
        deq_q = current_qparams
        deq_out = model.dequant(current)
        self._write_one_intermediate_float(EXPORT_DIR, next_layer_id, deq_out)

        if deq_in.ndim != 2 or deq_out.ndim != 2:
            raise RuntimeError(
                f"Dequant layer expects 2D tensors here, got input={tuple(deq_in.shape)}, output={tuple(deq_out.shape)}"
            )

        in_n = int(deq_in.shape[0])
        in_c = int(deq_in.shape[1])
        in_h = 1
        in_w = 1

        out_n = int(deq_out.shape[0])
        out_c = int(deq_out.shape[1])
        out_h = 1
        out_w = 1

        self.layer_descs.append(LayerDesc(
            layer_id=next_layer_id,
            name="dequant",
            op_type=OP_DEQUANTIZE,
            op_name="dequantize",
            in_n=in_n, in_c=in_c, in_h=in_h, in_w=in_w,
            out_n=out_n, out_c=out_c, out_h=out_h, out_w=out_w,
            k_h=0, k_w=0, s_h=0, s_w=0, p_h=0, p_w=0,
            has_relu=0,
            has_bias=0,
            input_dtype=DTYPE_INT8,
            output_dtype=DTYPE_FLOAT32,
            weight_offset=0,
            weight_bytes=0,
            bias_offset=0,
            bias_bytes=0,
            quant_record_index=len(self.quant_descs),
        ))
        self.quant_descs.append(QuantDesc(
            layer_id=next_layer_id,
            input_qscheme=deq_q["qscheme_code"],
            weight_qscheme=QSCHEME_NONE,
            output_qscheme=QSCHEME_NONE,
            input_scale=float(deq_q["scale"]),
            input_zero_point=int(deq_q["zero_point"]),
            output_scale=0.0,
            output_zero_point=0,
            weight_scale_offset=0,
            weight_scale_count=0,
            weight_zp_offset=0,
            weight_zp_count=0,
            weight_scale_single=0.0,
            weight_zero_point_single=0,
            weight_axis=-1,
        ))
        layers_info_for_manifest.append({
            "layer_id": next_layer_id,
            "name": "dequant",
            "op_name": "dequantize",
        })

        logits_fp32 = deq_out.detach().cpu().to(torch.float32)

        # --------------------------------------------------------
        # 9) Build tile plan and write files
        # --------------------------------------------------------
        self._build_tile_plan()
        self._write_cfg_files(EXPORT_DIR)
        self._write_tile_plan_file(EXPORT_DIR)
        self._write_param_files(EXPORT_DIR)
        self._write_quant_aux_files(EXPORT_DIR)
        self._write_input_and_output(EXPORT_DIR, input_int8, quant0_q, logits_fp32)

        pred_idx = int(torch.argmax(logits_fp32, dim=1).item())
        probs = torch.softmax(logits_fp32, dim=1)
        confidence = float(probs[0, pred_idx].item())

        self.manifest["sample"] = {
            "sample_index": sample_index,
            "true_label": int(true_label),
            "pred_label": pred_idx,
            "confidence": confidence,
            "input_shape_before_quant": list(input_fp32.shape),
            "input_shape_after_quant": list(input_int8.shape),
        }
        self.manifest["layers"] = layers_info_for_manifest
        self.manifest["checkpoint"] = CKPT_PATH
        self.manifest["export_dir"] = EXPORT_DIR
        self.manifest["intermediate_files"] = self.intermediate_files
        self.manifest["tile_plan_records"] = len(self.tile_descs)
        self.manifest["raw_image_mode"] = str(raw_img.mode) if hasattr(raw_img, "mode") else "unknown"

        with open(os.path.join(EXPORT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)

        print("Export finished.")
        print(f"Export dir: {os.path.abspath(EXPORT_DIR)}")
        print(f"Checkpoint: {CKPT_PATH}")
        print(f"Sample index: {sample_index}")
        print(f"True label: {true_label}")
        print(f"Pred label: {pred_idx}")
        print(f"Confidence: {confidence:.6f}")
        print(f"Layers exported: {len(self.layer_descs)}")


if __name__ == "__main__":
    exporter = HardwarePackageExporter()
    exporter.export()