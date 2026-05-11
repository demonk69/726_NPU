import os
import math
import zlib
import struct
from typing import Dict, List, Tuple, Optional

import numpy as np


# ============================================================
# Fixed path
# ============================================================
HW_DIR = r"./hardware_package"


# ============================================================
# Fixed protocol constants (must match exporter)
# ============================================================
ENDIAN = "<"
HEADER_BYTES = 64
CFG_RECORD_BYTES = 128
QUANT_RECORD_BYTES = 128
TILE_RECORD_BYTES = 128

MAGIC_LAYER_CFG = b"LYRCFG01"
MAGIC_QUANT_CFG = b"QNTCFG01"
MAGIC_WEIGHT = b"WEIGHT01"
MAGIC_BIAS = b"BIAS0001"
MAGIC_INPUT = b"INPUT001"
MAGIC_GOLDEN = b"GOLDEN01"
MAGIC_LAYER_OUT = b"LYROUT01"
MAGIC_TILE_PLAN = b"TILPLN01"
MAGIC_WEIGHT_SCALES = b"WSCAL001"
MAGIC_WEIGHT_ZPS = b"WZPNT001"[:8]

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

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ============================================================
# Low-level helpers
# ============================================================
def crc32_u32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def dtype_code_to_numpy(dtype_code: int):
    if dtype_code == DTYPE_INT8:
        return np.int8
    if dtype_code == DTYPE_UINT8:
        return np.uint8
    if dtype_code == DTYPE_INT32:
        return np.int32
    if dtype_code == DTYPE_FLOAT32:
        return np.float32
    raise ValueError(f"Unsupported dtype_code={dtype_code}")


def dtype_code_to_name(dtype_code: int) -> str:
    return {
        DTYPE_NONE: "none",
        DTYPE_INT8: "int8",
        DTYPE_UINT8: "uint8",
        DTYPE_INT32: "int32",
        DTYPE_FLOAT32: "float32",
    }.get(dtype_code, f"unknown({dtype_code})")


def op_type_to_name(op_type: int) -> str:
    return {
        OP_QUANTIZE: "quantize",
        OP_CONV_RELU: "conv_relu",
        OP_MAXPOOL: "maxpool",
        OP_AVGPOOL: "avgpool",
        OP_FLATTEN: "flatten",
        OP_LINEAR: "linear",
        OP_DEQUANTIZE: "dequantize",
    }.get(op_type, f"unknown({op_type})")


def qscheme_to_name(qscheme: int) -> str:
    return {
        QSCHEME_NONE: "none",
        QSCHEME_PER_TENSOR_AFFINE: "per_tensor_affine",
        QSCHEME_PER_CHANNEL_AFFINE: "per_channel_affine",
        QSCHEME_PER_TENSOR_SYMMETRIC: "per_tensor_symmetric",
        QSCHEME_PER_CHANNEL_SYMMETRIC: "per_channel_symmetric",
    }.get(qscheme, f"unknown({qscheme})")


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


def clamp_to_dtype_range(x: np.ndarray, dtype_code: int) -> np.ndarray:
    if dtype_code == DTYPE_INT8:
        return np.clip(x, -128, 127)
    if dtype_code == DTYPE_UINT8:
        return np.clip(x, 0, 255)
    if dtype_code == DTYPE_INT32:
        return np.clip(x, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    raise ValueError(f"Clamp not defined for dtype_code={dtype_code}")


# ============================================================
# Bin readers
# ============================================================
def read_bin_with_header(path: str, expected_magic: Optional[bytes] = None) -> Tuple[Dict[str, int], bytes]:
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < HEADER_BYTES:
        raise RuntimeError(f"File too small for header: {path}")

    header = data[:HEADER_BYTES]
    payload = data[HEADER_BYTES:]

    magic, version, header_bytes, payload_bytes, total_bytes, payload_crc, reserved0, extra0, extra1, extra2, extra3, _ = struct.unpack(
        ENDIAN + "8sIIQQIIIIII8s",
        header,
    )

    info = {
        "version": version,
        "header_bytes": header_bytes,
        "payload_bytes": payload_bytes,
        "total_bytes": total_bytes,
        "payload_crc": payload_crc,
        "reserved0": reserved0,
        "extra0": extra0,
        "extra1": extra1,
        "extra2": extra2,
        "extra3": extra3,
    }

    print(f"[READ] {os.path.basename(path)}")
    print(f"       magic={magic}, version={version}, header_bytes={header_bytes}, payload_bytes={payload_bytes}, total_bytes={total_bytes}")
    if expected_magic is not None and magic != expected_magic:
        print(f"[WARN] magic mismatch: expected={expected_magic!r}, got={magic!r}")
    if header_bytes != HEADER_BYTES:
        print(f"[WARN] header_bytes mismatch: expected={HEADER_BYTES}, got={header_bytes}")
    if payload_bytes != len(payload):
        print(f"[WARN] payload_bytes mismatch: header={payload_bytes}, actual={len(payload)}")
    actual_crc = crc32_u32(payload)
    if actual_crc != payload_crc:
        print(f"[WARN] payload CRC mismatch: header={payload_crc:#010x}, actual={actual_crc:#010x}")

    return info, payload


# ============================================================
# Protocol record parsing
# ============================================================
def parse_layer_cfg(payload: bytes) -> List[Dict[str, int]]:
    if len(payload) % CFG_RECORD_BYTES != 0:
        raise RuntimeError(f"layer_cfg payload size not multiple of {CFG_RECORD_BYTES}: {len(payload)}")

    layers = []
    n = len(payload) // CFG_RECORD_BYTES
    fmt = ENDIAN + "32I"
    size = struct.calcsize(fmt)
    assert size == 128

    for i in range(n):
        rec = payload[i * CFG_RECORD_BYTES:(i + 1) * CFG_RECORD_BYTES]
        vals = struct.unpack(fmt, rec)
        layers.append({
            "layer_id": vals[0],
            "op_type": vals[1],
            "in_n": vals[2], "in_c": vals[3], "in_h": vals[4], "in_w": vals[5],
            "out_n": vals[6], "out_c": vals[7], "out_h": vals[8], "out_w": vals[9],
            "k_h": vals[10], "k_w": vals[11],
            "s_h": vals[12], "s_w": vals[13],
            "p_h": vals[14], "p_w": vals[15],
            "has_relu": vals[16],
            "has_bias": vals[17],
            "input_dtype": vals[18],
            "output_dtype": vals[19],
            "weight_offset": vals[20],
            "weight_bytes": vals[21],
            "bias_offset": vals[22],
            "bias_bytes": vals[23],
            "quant_record_index": vals[24],
        })
    return layers


def parse_quant_cfg(payload: bytes) -> List[Dict[str, float]]:
    if len(payload) % QUANT_RECORD_BYTES != 0:
        raise RuntimeError(f"quant_cfg payload size not multiple of {QUANT_RECORD_BYTES}: {len(payload)}")

    qs = []
    fmt = ENDIAN + "IIIIfIfIIIIIfiiII"
    size = struct.calcsize(fmt)
    assert size == 68

    n = len(payload) // QUANT_RECORD_BYTES
    for i in range(n):
        rec = payload[i * QUANT_RECORD_BYTES:(i + 1) * QUANT_RECORD_BYTES]
        vals = struct.unpack(fmt, rec[:size])
        qs.append({
            "layer_id": vals[0],
            "input_qscheme": vals[1],
            "weight_qscheme": vals[2],
            "output_qscheme": vals[3],
            "input_scale": vals[4],
            "input_zero_point": vals[5],
            "output_scale": vals[6],
            "output_zero_point": vals[7],
            "weight_scale_offset": vals[8],
            "weight_scale_count": vals[9],
            "weight_zp_offset": vals[10],
            "weight_zp_count": vals[11],
            "weight_scale_single": vals[12],
            "weight_zero_point_single": vals[13],
            "weight_axis": vals[14],
            "reserved0": vals[15],
            "reserved1": vals[16],
        })
    return qs


def parse_tile_plan(payload: bytes) -> List[Dict[str, int]]:
    if len(payload) % TILE_RECORD_BYTES != 0:
        raise RuntimeError(f"tile_plan payload size not multiple of {TILE_RECORD_BYTES}: {len(payload)}")

    tiles = []
    fmt = ENDIAN + "32I"
    size = struct.calcsize(fmt)
    assert size == 128

    n = len(payload) // TILE_RECORD_BYTES
    for i in range(n):
        rec = payload[i * TILE_RECORD_BYTES:(i + 1) * TILE_RECORD_BYTES]
        vals = struct.unpack(fmt, rec)
        tiles.append({
            "layer_id": vals[0],
            "op_type": vals[1],
            "tile_id": vals[2],
            "group_id": vals[3],
            "array_dim": vals[4],
            "flags": vals[5],
            "oh_start": vals[6],
            "ow_start": vals[7],
            "oh_len": vals[8],
            "ow_len": vals[9],
            "oc_start": vals[10],
            "oc_len": vals[11],
            "k_start": vals[12],
            "k_len": vals[13],
            "m_dim": vals[14],
            "n_dim": vals[15],
            "k_dim": vals[16],
            "input_bytes": vals[17],
            "weight_bytes": vals[18],
            "output_bytes": vals[19],
        })
    return tiles


# ============================================================
# Tensor readers from blobs
# ============================================================
def read_array_from_payload(payload: bytes, offset: int, num_elems: int, np_dtype) -> np.ndarray:
    elem_bytes = np.dtype(np_dtype).itemsize
    start = offset
    end = offset + num_elems * elem_bytes
    if end > len(payload):
        raise RuntimeError(f"read_array out of range: offset={offset}, num_elems={num_elems}, dtype={np_dtype}, payload_len={len(payload)}")
    return np.frombuffer(payload[start:end], dtype=np_dtype).copy()


def read_weight_for_layer(layer: Dict[str, int], weight_payload: bytes) -> Optional[np.ndarray]:
    if layer["weight_bytes"] == 0:
        return None

    num_elems = layer["weight_bytes"]  # int8 权重，每个元素 1 byte
    arr = read_array_from_payload(weight_payload, layer["weight_offset"], num_elems, np.int8)

    if layer["op_type"] == OP_CONV_RELU:
        out_c = layer["out_c"]
        in_c = layer["in_c"]
        k_h = layer["k_h"]
        k_w = layer["k_w"]
        expected = out_c * in_c * k_h * k_w
        if arr.size != expected:
            raise RuntimeError(f"Conv weight size mismatch: got={arr.size}, expected={expected}, layer_id={layer['layer_id']}")
        return arr.reshape(out_c, in_c, k_h, k_w)

    if layer["op_type"] == OP_LINEAR:
        out_c = layer["out_c"]
        in_features = layer["in_c"] * layer["in_h"] * layer["in_w"]
        expected = out_c * in_features
        if arr.size != expected:
            raise RuntimeError(f"Linear weight size mismatch: got={arr.size}, expected={expected}, layer_id={layer['layer_id']}")
        return arr.reshape(out_c, in_features)

    raise RuntimeError(f"Unexpected weighted layer op_type={layer['op_type']}")


def read_bias_for_layer(layer: Dict[str, int], bias_payload: bytes) -> Optional[np.ndarray]:
    if layer["bias_bytes"] == 0:
        return None
    num_elems = layer["bias_bytes"] // 4
    arr = read_array_from_payload(bias_payload, layer["bias_offset"], num_elems, np.int32)
    return arr


def read_weight_scales_for_layer(q: Dict[str, float], scales_payload: bytes) -> Optional[np.ndarray]:
    cnt = q["weight_scale_count"]
    if cnt == 0:
        return None
    return read_array_from_payload(scales_payload, q["weight_scale_offset"], cnt, np.float32)


def read_weight_zps_for_layer(q: Dict[str, float], zps_payload: bytes) -> Optional[np.ndarray]:
    cnt = q["weight_zp_count"]
    if cnt == 0:
        return None
    return read_array_from_payload(zps_payload, q["weight_zp_offset"], cnt, np.int32)


def load_layer_output_bin(layer_id: int, expected_shape: Tuple[int, ...], expected_dtype_code: int) -> np.ndarray:
    path = os.path.join(HW_DIR, f"layer_{layer_id:03d}_out.bin")
    _, payload = read_bin_with_header(path, MAGIC_LAYER_OUT)
    np_dtype = dtype_code_to_numpy(expected_dtype_code)
    arr = np.frombuffer(payload, dtype=np_dtype).copy()
    expected_numel = int(np.prod(expected_shape))
    if arr.size != expected_numel:
        raise RuntimeError(f"Golden layer output numel mismatch for layer {layer_id}: got={arr.size}, expected={expected_numel}")
    return arr.reshape(expected_shape)


# ============================================================
# Integer requant utilities
# ============================================================
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


def requantize_per_tensor(
    acc: np.ndarray,
    input_scale: float,
    weight_scale: float,
    output_scale: float,
    output_zero_point: int,
    out_dtype_code: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    real_scale = (input_scale * weight_scale) / output_scale
    if not (0.0 < real_scale < 1.0):
        raise RuntimeError(f"real_scale out of expected range (0,1): {real_scale}")

    multiplier, shift = quantize_multiplier_smaller_than_one(real_scale)
    acc64 = acc.astype(np.int64)
    scaled = saturating_rounding_doubling_high_mul(acc64, multiplier)
    scaled = rounding_divide_by_pot(scaled, shift)
    scaled = scaled + np.int64(output_zero_point)
    scaled = clamp_to_dtype_range(scaled, out_dtype_code).astype(dtype_code_to_numpy(out_dtype_code))

    meta = {
        "real_scale": real_scale,
        "multiplier": multiplier,
        "shift": shift,
    }
    return scaled, meta


def requantize_per_channel(
    acc: np.ndarray,
    input_scale: float,
    weight_scales: np.ndarray,
    output_scale: float,
    output_zero_point: int,
    out_dtype_code: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    oc = acc.shape[1]
    if weight_scales.shape[0] != oc:
        raise RuntimeError(f"weight_scales len mismatch: got={weight_scales.shape[0]}, expected out_channels={oc}")

    out = np.empty_like(acc, dtype=dtype_code_to_numpy(out_dtype_code))
    multipliers = np.zeros((oc,), dtype=np.int64)
    shifts = np.zeros((oc,), dtype=np.int32)
    real_scales = np.zeros((oc,), dtype=np.float64)

    for c in range(oc):
        rs = (input_scale * float(weight_scales[c])) / output_scale
        if not (0.0 < rs < 1.0):
            raise RuntimeError(f"real_scale out of expected range (0,1): layer channel {c}, value={rs}")
        m, s = quantize_multiplier_smaller_than_one(rs)
        multipliers[c] = m
        shifts[c] = s
        real_scales[c] = rs

        cur = acc[:, c].astype(np.int64)
        scaled = saturating_rounding_doubling_high_mul(cur, m)
        scaled = rounding_divide_by_pot(scaled, s)
        scaled = scaled + np.int64(output_zero_point)
        scaled = clamp_to_dtype_range(scaled, out_dtype_code).astype(dtype_code_to_numpy(out_dtype_code))
        out[:, c] = scaled

    meta = {
        "real_scales": real_scales,
        "multipliers": multipliers,
        "shifts": shifts,
    }
    return out, meta


# ============================================================
# Core kernels (NumPy only)
# ============================================================
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


def relu_quantized(x: np.ndarray, zero_point: int, out_dtype_code: int) -> np.ndarray:
    x64 = x.astype(np.int64, copy=False)
    x64 = np.maximum(x64, np.int64(zero_point))
    x64 = clamp_to_dtype_range(x64, out_dtype_code)
    return x64.astype(dtype_code_to_numpy(out_dtype_code))


def maxpool2d_integer(x: np.ndarray, k_h: int, k_w: int, s_h: int, s_w: int) -> np.ndarray:
    n, c, h, w = x.shape
    out_h = (h - k_h) // s_h + 1
    out_w = (w - k_w) // s_w + 1
    out = np.empty((n, c, out_h, out_w), dtype=x.dtype)

    for oh in range(out_h):
        hs = oh * s_h
        for ow in range(out_w):
            ws = ow * s_w
            patch = x[:, :, hs:hs + k_h, ws:ws + k_w]
            out[:, :, oh, ow] = np.max(patch, axis=(2, 3))
    return out


def avgpool_integer_same_qparams(x: np.ndarray, k_h: int, k_w: int, s_h: int, s_w: int, out_dtype_code: int) -> np.ndarray:
    n, c, h, w = x.shape
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

    out = clamp_to_dtype_range(out, out_dtype_code).astype(dtype_code_to_numpy(out_dtype_code))
    return out


def flatten_nchw_to_nc(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


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


def dequantize_to_float(x_q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (x_q.astype(np.float32) - np.float32(zero_point)) * np.float32(scale)


# ============================================================
# Compare helpers
# ============================================================
def first_mismatch_float(a: np.ndarray, b: np.ndarray, atol: float = 1e-6):
    diff = np.abs(a - b) > atol
    if not np.any(diff):
        return None
    idx = np.argwhere(diff)[0]
    idx_t = tuple(int(v) for v in idx)
    return idx_t, a[idx_t], b[idx_t], float(abs(a[idx_t] - b[idx_t]))


def compare_and_stop(layer: Dict[str, int], actual: np.ndarray, expected: np.ndarray):
    layer_id = layer["layer_id"]
    op_name = op_type_to_name(layer["op_type"])
    dtype_code = layer["output_dtype"]

    print(f"[CHECK] layer_id={layer_id:03d}, op={op_name}, shape={actual.shape}, dtype={dtype_code_to_name(dtype_code)}")

    if dtype_code in (DTYPE_INT8, DTYPE_UINT8, DTYPE_INT32):
        diff = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
        neq = diff != 0
        mismatch_count = int(np.count_nonzero(neq))
        if mismatch_count > 0:
            idx = tuple(int(v) for v in np.argwhere(neq)[0])
            act = actual[idx]
            exp = expected[idx]
            delta = int(np.int64(act) - np.int64(exp))
            print(f"[FAIL] First mismatch at layer {layer_id}, idx={idx}, actual={act}, expected={exp}, delta={delta}")
            print(f"[FAIL] mismatch_count={mismatch_count} / {actual.size}")
            raise RuntimeError(f"Layer {layer_id} mismatch")
    elif dtype_code == DTYPE_FLOAT32:
        diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
        mismatch_count = int(np.count_nonzero(diff > 1e-6))
        if mismatch_count > 0:
            idx = tuple(int(v) for v in np.argwhere(diff > 1e-6)[0])
            act = actual[idx]
            exp = expected[idx]
            err = float(abs(act - exp))
            print(f"[FAIL] First mismatch at layer {layer_id}, idx={idx}, actual={act}, expected={exp}, abs_err={err}")
            print(f"[FAIL] mismatch_count={mismatch_count} / {actual.size}")
            raise RuntimeError(f"Layer {layer_id} mismatch")
    else:
        raise RuntimeError(f"Unsupported compare dtype_code={dtype_code}")

    print(f"[PASS] layer_id={layer_id:03d}")


# ============================================================
# Main replay engine
# ============================================================
class ReplayEngine:
    def __init__(self, hw_dir: str):
        self.hw_dir = hw_dir

        self.layer_cfg_payload = None
        self.quant_cfg_payload = None
        self.weight_payload = None
        self.bias_payload = None
        self.weight_scales_payload = None
        self.weight_zps_payload = None
        self.input_payload = None
        self.golden_payload = None
        self.tile_plan_payload = None

        self.layers: List[Dict[str, int]] = []
        self.quants: List[Dict[str, float]] = []
        self.tiles: List[Dict[str, int]] = []

    def load_all(self):
        _, self.layer_cfg_payload = read_bin_with_header(os.path.join(self.hw_dir, "layer_cfg.bin"), MAGIC_LAYER_CFG)
        _, self.quant_cfg_payload = read_bin_with_header(os.path.join(self.hw_dir, "quant_cfg.bin"), MAGIC_QUANT_CFG)
        _, self.weight_payload = read_bin_with_header(os.path.join(self.hw_dir, "weight.bin"), MAGIC_WEIGHT)
        _, self.bias_payload = read_bin_with_header(os.path.join(self.hw_dir, "bias.bin"), MAGIC_BIAS)
        _, self.weight_scales_payload = read_bin_with_header(os.path.join(self.hw_dir, "weight_scales.bin"), MAGIC_WEIGHT_SCALES)
        _, self.weight_zps_payload = read_bin_with_header(os.path.join(self.hw_dir, "weight_zero_points.bin"), None)
        _, self.input_payload = read_bin_with_header(os.path.join(self.hw_dir, "input.bin"), MAGIC_INPUT)
        _, self.golden_payload = read_bin_with_header(os.path.join(self.hw_dir, "golden_output.bin"), MAGIC_GOLDEN)
        _, self.tile_plan_payload = read_bin_with_header(os.path.join(self.hw_dir, "tile_plan.bin"), MAGIC_TILE_PLAN)

        self.layers = parse_layer_cfg(self.layer_cfg_payload)
        self.quants = parse_quant_cfg(self.quant_cfg_payload)
        self.tiles = parse_tile_plan(self.tile_plan_payload)

        if len(self.layers) != len(self.quants):
            raise RuntimeError(f"layer_cfg count != quant_cfg count: {len(self.layers)} vs {len(self.quants)}")

        print(f"[INFO] Loaded {len(self.layers)} layers and {len(self.tiles)} tile records")

    def get_quant(self, layer: Dict[str, int]) -> Dict[str, float]:
        idx = layer["quant_record_index"]
        if idx < 0 or idx >= len(self.quants):
            raise RuntimeError(f"Invalid quant_record_index={idx} for layer_id={layer['layer_id']}")
        q = self.quants[idx]
        if q["layer_id"] != layer["layer_id"]:
            print(f"[WARN] quant layer_id mismatch: layer_cfg={layer['layer_id']}, quant_cfg={q['layer_id']}")
        return q

    def load_input(self) -> np.ndarray:
        layer0 = self.layers[0]
        shape = (layer0["out_n"], layer0["out_c"], layer0["out_h"], layer0["out_w"])
        dtype_code = layer0["output_dtype"]
        np_dtype = dtype_code_to_numpy(dtype_code)
        arr = np.frombuffer(self.input_payload, dtype=np_dtype).copy()
        expected_numel = int(np.prod(shape))
        if arr.size != expected_numel:
            raise RuntimeError(f"input.bin numel mismatch: got={arr.size}, expected={expected_numel}")
        return arr.reshape(shape)

    def load_final_golden(self) -> np.ndarray:
        arr = np.frombuffer(self.golden_payload, dtype=np.float32).copy()
        last = self.layers[-1]
        shape = (last["out_n"], last["out_c"])
        expected_numel = int(np.prod(shape))
        if arr.size != expected_numel:
            raise RuntimeError(f"golden_output.bin numel mismatch: got={arr.size}, expected={expected_numel}")
        return arr.reshape(shape)

    def get_tiles_for_layer(self, layer_id: int) -> List[Dict[str, int]]:
        return [t for t in self.tiles if t["layer_id"] == layer_id]

    def execute_conv_tiled(self, layer: Dict[str, int], q: Dict[str, float], current: np.ndarray) -> np.ndarray:
        tiles = self.get_tiles_for_layer(layer["layer_id"])
        if not tiles:
            raise RuntimeError(f"No tile records for conv layer {layer['layer_id']}")
        weight = read_weight_for_layer(layer, self.weight_payload)
        bias = read_bias_for_layer(layer, self.bias_payload)
        weight_scales = read_weight_scales_for_layer(q, self.weight_scales_payload)
        weight_zps = read_weight_zps_for_layer(q, self.weight_zps_payload)
        if weight is None or bias is None or weight_scales is None or weight_zps is None:
            raise RuntimeError(f"Missing conv params for layer {layer['layer_id']}")
        out = np.zeros((layer['out_n'], layer['out_c'], layer['out_h'], layer['out_w']), dtype=dtype_code_to_numpy(layer['output_dtype']))
        psums: Dict[int, np.ndarray] = {}
        xpad = pad_nchw_int(current, layer['p_h'], layer['p_w'], int(q['input_zero_point']))
        khkw = layer['k_h'] * layer['k_w']
        for tile in tiles:
            oc0 = tile['oc_start']; oc1 = oc0 + tile['oc_len']
            oh0 = tile['oh_start']; oh1 = oh0 + tile['oh_len']
            ow0 = tile['ow_start']; ow1 = ow0 + tile['ow_len']
            cin0 = tile['k_start'] // khkw
            cin_len = tile['k_len'] // khkw
            cin1 = cin0 + cin_len
            key = tile['group_id']
            if key not in psums:
                psums[key] = np.zeros((layer['out_n'], tile['oc_len'], tile['oh_len'], tile['ow_len']), dtype=np.int32)
            cur_psum = psums[key]
            for n in range(layer['out_n']): #先处理第N张图片，也就是第N个样本
                for oc_rel, oc in enumerate(range(oc0, oc1)):   #遍历当前tile负责的输出通道
                    for oh_rel, oh in enumerate(range(oh0, oh1)):   
                        hs = oh * layer['s_h']  #根据输出高度坐标，计算感受野左上角的高度起点
                        for ow_rel, ow in enumerate(range(ow0, ow1)):
                            ws = ow * layer['s_w']
                            acc = 0
                            for ic in range(cin0, cin1):    #遍历每一个输出通道
                                for kh in range(layer['k_h']):
                                    for kw in range(layer['k_w']):
                                        xv = int(xpad[n, ic, hs + kh, ws + kw]) - int(q['input_zero_point'])
                                        wv = int(weight[oc, ic, kh, kw]) - int(weight_zps[oc])
                                        acc += xv * wv
                            if cin0 == 0:
                                acc += int(bias[oc])
                            cur_psum[n, oc_rel, oh_rel, ow_rel] += np.int32(acc)
            if tile['flags'] & 2:
                rq, _ = requantize_per_channel(
                    cur_psum,
                    input_scale=float(q['input_scale']),
                    weight_scales=weight_scales[oc0:oc1],
                    output_scale=float(q['output_scale']),
                    output_zero_point=int(q['output_zero_point']),
                    out_dtype_code=layer['output_dtype'],
                )
                if layer['has_relu']:
                    rq = relu_quantized(rq, int(q['output_zero_point']), layer['output_dtype'])
                out[:, oc0:oc1, oh0:oh1, ow0:ow1] = rq
                del psums[key]
        if psums:
            raise RuntimeError(f"Unflushed conv psums at layer {layer['layer_id']}")
        return out

    def execute_linear_tiled(self, layer: Dict[str, int], q: Dict[str, float], current: np.ndarray) -> np.ndarray:
        tiles = self.get_tiles_for_layer(layer['layer_id'])
        if not tiles:
            raise RuntimeError(f"No tile records for linear layer {layer['layer_id']}")
        weight = read_weight_for_layer(layer, self.weight_payload)
        bias = read_bias_for_layer(layer, self.bias_payload)
        weight_scales = read_weight_scales_for_layer(q, self.weight_scales_payload)
        weight_zps = read_weight_zps_for_layer(q, self.weight_zps_payload)
        if weight is None or bias is None or weight_scales is None or weight_zps is None:
            raise RuntimeError(f"Missing linear params for layer {layer['layer_id']}")
        out = np.zeros((layer['out_n'], layer['out_c']), dtype=dtype_code_to_numpy(layer['output_dtype']))
        psums: Dict[int, np.ndarray] = {}
        x = current.reshape(current.shape[0], -1)
        for tile in tiles:
            oc0 = tile['oc_start']; oc1 = oc0 + tile['oc_len']
            k0 = tile['k_start']; k1 = k0 + tile['k_len']
            key = tile['group_id']
            if key not in psums:
                psums[key] = np.zeros((layer['out_n'], tile['oc_len']), dtype=np.int32)
            cur_psum = psums[key]
            x_part = x[:, k0:k1].astype(np.int32) - np.int32(q['input_zero_point'])
            w_part = weight[oc0:oc1, k0:k1].astype(np.int32) - weight_zps[oc0:oc1].astype(np.int32)[:, None]
            cur_psum += x_part @ w_part.T
            if k0 == 0:
                cur_psum += bias[oc0:oc1].astype(np.int32)[None, :]
            if tile['flags'] & 2:
                rq, _ = requantize_per_channel(
                    cur_psum[:, :, None, None],
                    input_scale=float(q['input_scale']),
                    weight_scales=weight_scales[oc0:oc1],
                    output_scale=float(q['output_scale']),
                    output_zero_point=int(q['output_zero_point']),
                    out_dtype_code=layer['output_dtype'],
                )
                out[:, oc0:oc1] = rq[:, :, 0, 0]
                del psums[key]
        if psums:
            raise RuntimeError(f"Unflushed linear psums at layer {layer['layer_id']}")
        return out

    def replay(self):
        self.load_all()

        current = self.load_input()
        print(f"[INFO] Starting from input.bin, shape={current.shape}, dtype={current.dtype}")

        # layer 0: quantize
        layer0 = self.layers[0]
        golden0 = load_layer_output_bin(0, current.shape, layer0["output_dtype"])
        print(f"[LAYER] id=000 op=quantize")
        print(f"        input_shape=(1,3,32,32) [implicit from package design]")
        print(f"        output_shape={current.shape}")
        q0 = self.get_quant(layer0)
        print(f"        output_qscheme={qscheme_to_name(q0['output_qscheme'])}, output_scale={q0['output_scale']}, output_zero_point={q0['output_zero_point']}")
        compare_and_stop(layer0, current, golden0)

        for li in range(1, len(self.layers)):
            layer = self.layers[li]
            q = self.get_quant(layer)
            op_name = op_type_to_name(layer["op_type"])

            in_shape = (layer["in_n"], layer["in_c"], layer["in_h"], layer["in_w"])
            if layer["op_type"] in (OP_LINEAR, OP_DEQUANTIZE, OP_FLATTEN):
                out_shape = (layer["out_n"], layer["out_c"])
            else:
                out_shape = (layer["out_n"], layer["out_c"], layer["out_h"], layer["out_w"])

            print(f"[LAYER] id={layer['layer_id']:03d} op={op_name}")
            print(f"        in_shape_cfg={in_shape}, out_shape_cfg={out_shape}")
            print(f"        input_dtype={dtype_code_to_name(layer['input_dtype'])}, output_dtype={dtype_code_to_name(layer['output_dtype'])}")
            print(f"        input_qscheme={qscheme_to_name(q['input_qscheme'])}, weight_qscheme={qscheme_to_name(q['weight_qscheme'])}, output_qscheme={qscheme_to_name(q['output_qscheme'])}")
            print(f"        input_scale={q['input_scale']}, input_zero_point={q['input_zero_point']}")
            print(f"        output_scale={q['output_scale']}, output_zero_point={q['output_zero_point']}")
            print(f"        weight_offset={layer['weight_offset']}, weight_bytes={layer['weight_bytes']}, bias_offset={layer['bias_offset']}, bias_bytes={layer['bias_bytes']}")

            if layer["op_type"] == OP_CONV_RELU:
                tiles = self.get_tiles_for_layer(layer['layer_id'])
                print(f"        conv kernel=({layer['k_h']},{layer['k_w']}), stride=({layer['s_h']},{layer['s_w']}), padding=({layer['p_h']},{layer['p_w']})")
                print(f"        tile_records={len(tiles)}")
                actual = self.execute_conv_tiled(layer, q, current)

            elif layer["op_type"] == OP_MAXPOOL:
                print(f"        pool kernel=({layer['k_h']},{layer['k_w']}), stride=({layer['s_h']},{layer['s_w']})")
                actual = maxpool2d_integer(current, layer['k_h'], layer['k_w'], layer['s_h'], layer['s_w'])

            elif layer["op_type"] == OP_AVGPOOL:
                print(f"        avgpool kernel=({layer['k_h']},{layer['k_w']}), stride=({layer['s_h']},{layer['s_w']})")
                actual = avgpool_integer_same_qparams(
                    current,
                    layer['k_h'],
                    layer['k_w'],
                    layer['s_h'],
                    layer['s_w'],
                    layer['output_dtype'],
                )

            elif layer["op_type"] == OP_FLATTEN:
                actual = flatten_nchw_to_nc(current)
                print(f"        flatten output_shape={actual.shape}")

            elif layer["op_type"] == OP_LINEAR:
                tiles = self.get_tiles_for_layer(layer['layer_id'])
                print(f"        linear tile_records={len(tiles)}")
                actual = self.execute_linear_tiled(layer, q, current)

            elif layer["op_type"] == OP_DEQUANTIZE:
                actual = dequantize_to_float(current, float(q['input_scale']), int(q['input_zero_point']))
                print(f"        dequant scale={q['input_scale']}, zero_point={q['input_zero_point']}")

            else:
                raise RuntimeError(f"Unsupported layer op_type={layer['op_type']} at layer_id={layer['layer_id']}")

            golden = load_layer_output_bin(layer['layer_id'], actual.shape, layer['output_dtype'])
            compare_and_stop(layer, actual, golden)
            current = actual

        final_golden = self.load_final_golden()
        print("[FINAL] comparing final output with golden_output.bin")
        mm = first_mismatch_float(current.astype(np.float32), final_golden.astype(np.float32), atol=1e-6)
        if mm is not None:
            idx, act, exp, err = mm
            print(f"[FAIL] final logits mismatch at idx={idx}, actual={act}, expected={exp}, abs_err={err}")
            raise RuntimeError("Final logits mismatch")

        pred = int(np.argmax(current[0]))
        logits = current[0].astype(np.float64)
        exps = np.exp(logits - np.max(logits))
        conf = float(exps[pred] / np.sum(exps))
        print("[PASS] final logits match golden_output.bin")
        print(f"[RESULT] pred={pred} ({CLASSES[pred]}), pseudo_confidence={conf:.6f}")
        print("Replay finished successfully.")


if __name__ == "__main__":
    engine = ReplayEngine(HW_DIR)
    engine.replay()