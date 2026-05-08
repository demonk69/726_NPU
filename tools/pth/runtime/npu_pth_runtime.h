#ifndef NPU_PTH_RUNTIME_H
#define NPU_PTH_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#define NPU_PTH_REG_CTRL             0x00u
#define NPU_PTH_REG_STATUS           0x04u
#define NPU_PTH_REG_M_DIM            0x10u
#define NPU_PTH_REG_N_DIM            0x14u
#define NPU_PTH_REG_K_DIM            0x18u
#define NPU_PTH_REG_W_ADDR           0x20u
#define NPU_PTH_REG_A_ADDR           0x24u
#define NPU_PTH_REG_R_ADDR           0x28u
#define NPU_PTH_REG_CONV_IFM_SHAPE   0x80u
#define NPU_PTH_REG_CONV_CHANNELS    0x84u
#define NPU_PTH_REG_CONV_KERNEL      0x88u
#define NPU_PTH_REG_CONV_OUT_SHAPE   0x8cu
#define NPU_PTH_REG_CONV_STRIDE_PAD  0x90u
#define NPU_PTH_REG_CONV_DILATION    0x94u
#define NPU_PTH_REG_BIAS_ADDR        0x98u
#define NPU_PTH_REG_QUANT_CFG        0x9cu

#define NPU_PTH_STATUS_BUSY          0x00000001u
#define NPU_PTH_STATUS_DONE          0x00000002u
#define NPU_PTH_STATUS_ERROR         0x00000004u

#define NPU_PTH_DEFAULT_TIMEOUT      0x01000000u

typedef enum npu_pth_op {
    NPU_PTH_OP_CONV2D = 1,
    NPU_PTH_OP_MAXPOOL2D = 2,
    NPU_PTH_OP_ADAPTIVE_AVGPOOL2D = 3,
    NPU_PTH_OP_FLATTEN = 4,
    NPU_PTH_OP_LINEAR = 5,
} npu_pth_op_t;

typedef enum npu_pth_exec {
    NPU_PTH_EXEC_CPU = 0,
    NPU_PTH_EXEC_NPU_DIRECT = 1,
} npu_pth_exec_t;

typedef struct npu_pth_shape {
    uint32_t rank;
    uint32_t n;
    uint32_t c;
    uint32_t h;
    uint32_t w;
} npu_pth_shape_t;

#define NPU_PTH_SHAPE4(n_, c_, h_, w_) {4u, (n_), (c_), (h_), (w_)}
#define NPU_PTH_SHAPE2(n_, c_)        {2u, (n_), (c_), 1u, 1u}

typedef struct npu_pth_layer {
    const char *name;
    npu_pth_op_t op;
    npu_pth_exec_t exec;
    npu_pth_shape_t input;
    npu_pth_shape_t output;

    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t dilation_h;
    uint32_t dilation_w;

    uint32_t m_dim;
    uint32_t n_dim;
    uint32_t k_dim;
    uint32_t w_addr;
    uint32_t bias_addr;
    uint32_t w_bytes;
    uint32_t bias_bytes;
    uint32_t ctrl;
    uint32_t conv_ifm_shape;
    uint32_t conv_channels;
    uint32_t conv_kernel;
    uint32_t conv_out_shape;
    uint32_t conv_stride_pad;
    uint32_t conv_dilation;
    uint32_t quant_cfg;

    const int32_t *requant_multiplier;
    const uint8_t *requant_shift;
    uint32_t requant_count;
    int32_t output_zero_point;
} npu_pth_layer_t;

static inline void npu_pth_write32(uintptr_t npu_base, uint32_t offset, uint32_t value)
{
    *(volatile uint32_t *)(npu_base + offset) = value;
}

static inline uint32_t npu_pth_read32(uintptr_t npu_base, uint32_t offset)
{
    return *(volatile uint32_t *)(npu_base + offset);
}

static inline int8_t npu_pth_clamp_i8(int32_t value)
{
    if (value > 127) {
        return 127;
    }
    if (value < -128) {
        return -128;
    }
    return (int8_t)value;
}

static inline int64_t npu_pth_round_shift_i64(int64_t value, uint8_t shift)
{
    if (shift == 0u) {
        return value;
    }
    if (shift >= 63u) {
        return 0;
    }
    if (value >= 0) {
        value += (int64_t)1 << (shift - 1u);
    } else {
        value -= (int64_t)1 << (shift - 1u);
    }
    return value >> shift;
}

static inline int8_t npu_pth_requant_i8(const npu_pth_layer_t *layer, uint32_t channel, int32_t acc)
{
    int64_t scaled = acc;
    if (layer->requant_multiplier != 0 && layer->requant_shift != 0 && channel < layer->requant_count) {
        scaled = (int64_t)acc * (int64_t)layer->requant_multiplier[channel];
        scaled = npu_pth_round_shift_i64(scaled, layer->requant_shift[channel]);
    }
    scaled += layer->output_zero_point;
    return npu_pth_clamp_i8((int32_t)scaled);
}

static inline int npu_pth_run_conv_direct(
    uintptr_t npu_base,
    const npu_pth_layer_t *layer,
    uint32_t ifm_addr,
    uint32_t acc_addr,
    uint32_t timeout)
{
    if (layer == 0 || layer->op != NPU_PTH_OP_CONV2D || layer->exec != NPU_PTH_EXEC_NPU_DIRECT) {
        return -3;
    }

    npu_pth_write32(npu_base, NPU_PTH_REG_CTRL, 0u);
    npu_pth_write32(npu_base, NPU_PTH_REG_M_DIM, layer->m_dim);
    npu_pth_write32(npu_base, NPU_PTH_REG_N_DIM, layer->n_dim);
    npu_pth_write32(npu_base, NPU_PTH_REG_K_DIM, layer->k_dim);
    npu_pth_write32(npu_base, NPU_PTH_REG_W_ADDR, layer->w_addr);
    npu_pth_write32(npu_base, NPU_PTH_REG_A_ADDR, ifm_addr);
    npu_pth_write32(npu_base, NPU_PTH_REG_R_ADDR, acc_addr);
    npu_pth_write32(npu_base, NPU_PTH_REG_CONV_IFM_SHAPE, layer->conv_ifm_shape);
    npu_pth_write32(npu_base, NPU_PTH_REG_CONV_CHANNELS, layer->conv_channels);
    npu_pth_write32(npu_base, NPU_PTH_REG_CONV_KERNEL, layer->conv_kernel);
    npu_pth_write32(npu_base, NPU_PTH_REG_CONV_OUT_SHAPE, layer->conv_out_shape);
    npu_pth_write32(npu_base, NPU_PTH_REG_CONV_STRIDE_PAD, layer->conv_stride_pad);
    npu_pth_write32(npu_base, NPU_PTH_REG_CONV_DILATION, layer->conv_dilation);
    npu_pth_write32(npu_base, NPU_PTH_REG_BIAS_ADDR, layer->bias_addr);
    npu_pth_write32(npu_base, NPU_PTH_REG_QUANT_CFG, layer->quant_cfg);
    npu_pth_write32(npu_base, NPU_PTH_REG_CTRL, layer->ctrl);

    while (timeout-- != 0u) {
        uint32_t status = npu_pth_read32(npu_base, NPU_PTH_REG_STATUS);
        if ((status & NPU_PTH_STATUS_ERROR) != 0u) {
            return -2;
        }
        if ((status & NPU_PTH_STATUS_DONE) != 0u) {
            return 0;
        }
    }
    return -1;
}

static inline void npu_pth_requant_repack_acc_to_nchw_i8(
    const npu_pth_layer_t *layer,
    const int32_t *acc_mn,
    int8_t *dst_nchw)
{
    uint32_t batch = layer->output.n;
    uint32_t channels = layer->output.c;
    uint32_t height = layer->output.h;
    uint32_t width = layer->output.w;

    for (uint32_t n = 0; n < batch; ++n) {
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                uint32_t m = ((n * height + y) * width) + x;
                for (uint32_t c = 0; c < channels; ++c) {
                    uint32_t src_idx = m * channels + c;
                    uint32_t dst_idx = ((n * channels + c) * height + y) * width + x;
                    dst_nchw[dst_idx] = npu_pth_requant_i8(layer, c, acc_mn[src_idx]);
                }
            }
        }
    }
}

static inline void npu_pth_maxpool2d_i8_nchw(
    const npu_pth_layer_t *layer,
    const int8_t *src_nchw,
    int8_t *dst_nchw)
{
    uint32_t batch = layer->input.n;
    uint32_t channels = layer->input.c;
    uint32_t ih = layer->input.h;
    uint32_t iw = layer->input.w;
    uint32_t oh = layer->output.h;
    uint32_t ow = layer->output.w;

    for (uint32_t n = 0; n < batch; ++n) {
        for (uint32_t c = 0; c < channels; ++c) {
            for (uint32_t y = 0; y < oh; ++y) {
                for (uint32_t x = 0; x < ow; ++x) {
                    int32_t best = -128;
                    for (uint32_t ky = 0; ky < layer->kernel_h; ++ky) {
                        int32_t in_y = (int32_t)(y * layer->stride_h + ky * layer->dilation_h) - (int32_t)layer->pad_h;
                        if (in_y < 0 || in_y >= (int32_t)ih) {
                            continue;
                        }
                        for (uint32_t kx = 0; kx < layer->kernel_w; ++kx) {
                            int32_t in_x = (int32_t)(x * layer->stride_w + kx * layer->dilation_w) - (int32_t)layer->pad_w;
                            if (in_x < 0 || in_x >= (int32_t)iw) {
                                continue;
                            }
                            uint32_t src_idx = ((n * channels + c) * ih + (uint32_t)in_y) * iw + (uint32_t)in_x;
                            if ((int32_t)src_nchw[src_idx] > best) {
                                best = src_nchw[src_idx];
                            }
                        }
                    }
                    dst_nchw[((n * channels + c) * oh + y) * ow + x] = (int8_t)best;
                }
            }
        }
    }
}

static inline uint32_t npu_pth_floor_div_u32(uint32_t a, uint32_t b)
{
    return a / b;
}

static inline uint32_t npu_pth_ceil_div_u32(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
}

static inline void npu_pth_adaptive_avgpool2d_i8_nchw(
    const npu_pth_layer_t *layer,
    const int8_t *src_nchw,
    int8_t *dst_nchw)
{
    uint32_t batch = layer->input.n;
    uint32_t channels = layer->input.c;
    uint32_t ih = layer->input.h;
    uint32_t iw = layer->input.w;
    uint32_t oh = layer->output.h;
    uint32_t ow = layer->output.w;

    for (uint32_t n = 0; n < batch; ++n) {
        for (uint32_t c = 0; c < channels; ++c) {
            for (uint32_t y = 0; y < oh; ++y) {
                uint32_t y0 = npu_pth_floor_div_u32(y * ih, oh);
                uint32_t y1 = npu_pth_ceil_div_u32((y + 1u) * ih, oh);
                for (uint32_t x = 0; x < ow; ++x) {
                    uint32_t x0 = npu_pth_floor_div_u32(x * iw, ow);
                    uint32_t x1 = npu_pth_ceil_div_u32((x + 1u) * iw, ow);
                    int32_t sum = 0;
                    uint32_t count = 0;
                    for (uint32_t yy = y0; yy < y1; ++yy) {
                        for (uint32_t xx = x0; xx < x1; ++xx) {
                            sum += src_nchw[((n * channels + c) * ih + yy) * iw + xx];
                            ++count;
                        }
                    }
                    if (sum >= 0) {
                        sum += (int32_t)(count / 2u);
                    } else {
                        sum -= (int32_t)(count / 2u);
                    }
                    dst_nchw[((n * channels + c) * oh + y) * ow + x] = npu_pth_clamp_i8(sum / (int32_t)count);
                }
            }
        }
    }
}

static inline void npu_pth_linear_i8(
    const npu_pth_layer_t *layer,
    const int8_t *src,
    const int8_t *weight_out_in,
    const int32_t *bias,
    int8_t *dst)
{
    uint32_t batch = layer->input.n;
    uint32_t in_features = layer->input.c;
    uint32_t out_features = layer->output.c;

    for (uint32_t n = 0; n < batch; ++n) {
        for (uint32_t out_c = 0; out_c < out_features; ++out_c) {
            int32_t acc = bias ? bias[out_c] : 0;
            for (uint32_t in_c = 0; in_c < in_features; ++in_c) {
                int32_t a = src[n * in_features + in_c];
                int32_t w = weight_out_in[out_c * in_features + in_c];
                acc += a * w;
            }
            dst[n * out_features + out_c] = npu_pth_requant_i8(layer, out_c, acc);
        }
    }
}

static inline uint32_t npu_pth_argmax_i8(const int8_t *values, uint32_t count)
{
    uint32_t best = 0;
    for (uint32_t i = 1; i < count; ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return best;
}

#endif
