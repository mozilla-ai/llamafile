#include "models.h"

// Apertus 1.5 vision tokenizer (EMU3.5 IBQ), encode only: image -> grid of discrete codes
// ref: transformers Apertus1p5VisionTokenizerModel.encode
// activations are [W, H, C] like the input image

static const int APERTUS_VQ_N_GROUPS = 32;

// codes are argmax over the codebook logits, bound the size of the logits with this many positions per step
static const int64_t APERTUS_VQ_CODES_CHUNK = 1024;

// conv as a sum of matmuls, one per kernel tap, on [C, W, H] views of the input:
// uses the matmul kernels (the direct conv kernels are slow at 256+ channels) and needs no im2col buffer
ggml_tensor * clip_graph_apertus1p5v::conv2d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, int pad) const {
    const int64_t KW   = w->ne[0];
    const int64_t KH   = w->ne[1];
    const int64_t Cin  = w->ne[2];
    const int64_t Cout = w->ne[3];
    GGML_ASSERT(x->ne[2] == Cin);

    // [W, H, C] -> [C, W, H]
    ggml_tensor * xc = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 2, 0, 3));
    if (pad > 0) {
        xc = ggml_pad_ext(ctx0, xc, 0, 0, pad, pad, pad, pad, 0, 0);
    }
    const int64_t Wp = xc->ne[1];
    const int64_t Wo = (Wp - KW) / stride + 1;
    const int64_t Ho = (xc->ne[2] - KH) / stride + 1;

    // [KW, KH, Cin, Cout] -> [Cin, Cout, KW, KH]
    ggml_tensor * wt = KW * KH > 1 ? ggml_cont(ctx0, ggml_permute(ctx0, w, 2, 3, 0, 1)) : ggml_reshape_2d(ctx0, w, Cin, Cout);

    // stride 1: every tap reads a contiguous run of the padded input, the output is computed on the padded
    // width and cropped at the end; contiguous src1 keeps the fast matmul paths (tinyBLAS on CPU)
    // stride > 1: the taps are strided views, copied to be contiguous
    const int64_t n = stride == 1 ? Wp * Ho - (KW - 1) : Wo * Ho;
    ggml_tensor * out = nullptr;
    for (int64_t ky = 0; ky < KH; ky++) {
        for (int64_t kx = 0; kx < KW; kx++) {
            ggml_tensor * wk = KW * KH > 1 ? ggml_view_2d(ctx0, wt, Cin, Cout, wt->nb[1], kx * wt->nb[2] + ky * wt->nb[3]) : wt;
            ggml_tensor * xk;
            if (stride == 1) {
                xk = ggml_view_2d(ctx0, xc, Cin, n, xc->nb[1], (kx + ky * Wp) * xc->nb[1]);
            } else {
                xk = ggml_view_3d(ctx0, xc, Cin, Wo, Ho, stride * xc->nb[1], stride * xc->nb[2], kx * xc->nb[1] + ky * xc->nb[2]);
                xk = ggml_reshape_2d(ctx0, ggml_cont(ctx0, xk), Cin, n);
            }
            ggml_tensor * yk = ggml_mul_mat(ctx0, wk, xk); // [Cout, n]
            out = out ? ggml_add(ctx0, out, yk) : yk;
        }
    }
    if (b) {
        out = ggml_add(ctx0, out, b);
    }

    // [C, W, H] -> [W, H, C], dropping the padding columns of the stride 1 case
    const int64_t row = stride == 1 ? Wp : Wo;
    out = ggml_view_3d(ctx0, out, Cout, Wo, Ho, out->nb[1], row * out->nb[1], 0);
    return ggml_cont(ctx0, ggml_permute(ctx0, out, 2, 0, 1, 3));
}

ggml_tensor * clip_graph_apertus1p5v::group_norm(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) const {
    x = ggml_group_norm(ctx0, x, APERTUS_VQ_N_GROUPS, eps);
    x = ggml_mul(ctx0, x, ggml_reshape_3d(ctx0, w, 1, 1, w->ne[0]));
    x = ggml_add(ctx0, x, ggml_reshape_3d(ctx0, b, 1, 1, b->ne[0]));
    return x;
}

ggml_tensor * clip_graph_apertus1p5v::resblock(ggml_tensor * x, const clip_apertus_vq::resblock & r) const {
    ggml_tensor * h = ggml_silu(ctx0, group_norm(x, r.a.norm_w, r.a.norm_b));
    h = conv2d(h, r.a.conv_w, r.a.conv_b, 1, 1);
    h = ggml_silu(ctx0, group_norm(h, r.b.norm_w, r.b.norm_b));
    h = conv2d(h, r.b.conv_w, r.b.conv_b, 1, 1);
    if (r.nin_w) {
        x = conv2d(x, r.nin_w, r.nin_b, 1, 0);
    }
    return ggml_add(ctx0, x, h);
}

ggml_tensor * clip_graph_apertus1p5v::attnblock(ggml_tensor * x, const clip_apertus_vq::attnblock & a) const {
    const int64_t W = x->ne[0];
    const int64_t H = x->ne[1];
    const int64_t C = x->ne[2];
    const int64_t N = W * H;

    ggml_tensor * h = group_norm(x, a.norm_w, a.norm_b);
    ggml_tensor * q = conv2d(h, a.q_w, a.q_b, 1, 0);
    ggml_tensor * k = conv2d(h, a.k_w, a.k_b, 1, 0);
    ggml_tensor * v = conv2d(h, a.v_w, a.v_b, 1, 0);

    // [W, H, C] -> [C, N]
    q = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, q, 1, 2, 0, 3)), C, N);
    k = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, k, 1, 2, 0, 3)), C, N);
    // [W, H, C] -> [N, C]
    v = ggml_reshape_2d(ctx0, v, N, C);

    ggml_tensor * kq = ggml_mul_mat(ctx0, k, q); // [N_k, N_q]
    kq = ggml_soft_max_ext(ctx0, kq, nullptr, 1.0f / sqrtf((float) C), 0.0f);

    ggml_tensor * out = ggml_mul_mat(ctx0, v, kq); // [C, N_q]
    out = ggml_cont(ctx0, ggml_transpose(ctx0, out)); // [N_q, C]
    out = ggml_reshape_3d(ctx0, out, W, H, C);

    out = conv2d(out, a.o_w, a.o_b, 1, 0);
    return ggml_add(ctx0, x, out);
}

ggml_cgraph * clip_graph_apertus1p5v::build() {
    const auto & vq = model.apertus_vq;

    ggml_tensor * cur = build_inp_raw();
    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], cur->ne[1], cur->ne[2]);

    cur = conv2d(cur, vq.conv_in_w, vq.conv_in_b, 1, 1);
    cb(cur, "conv_in", -1);

    for (size_t is = 0; is < vq.stages.size(); is++) {
        const auto & st = vq.stages[is];
        for (size_t il = 0; il < st.res.size(); il++) {
            cur = resblock(cur, st.res[il]);
            if (!st.attn.empty()) {
                cur = attnblock(cur, st.attn[il]);
            }
        }
        if (st.down_w) {
            // asymmetric right/bottom padding, then stride 2
            cur = ggml_pad_ext(ctx0, cur, 0, 1, 0, 1, 0, 0, 0, 0);
            cur = conv2d(cur, st.down_w, st.down_b, 2, 0);
        }
        cb(cur, "stage_out", (int) is);
    }

    cur = resblock(cur, vq.mid_res0);
    cur = attnblock(cur, vq.mid_attn);
    cur = resblock(cur, vq.mid_res1);

    cur = ggml_silu(ctx0, group_norm(cur, vq.norm_out_w, vq.norm_out_b));
    cur = conv2d(cur, vq.conv_out_w, vq.conv_out_b, 1, 1);
    cur = conv2d(cur, vq.quant_conv_w, vq.quant_conv_b, 1, 0);
    cb(cur, "latent", -1);

    // IBQ: the code is the codebook row with the largest dot product, row-major over the grid
    const int64_t D = cur->ne[2];
    const int64_t N = cur->ne[0] * cur->ne[1];
    ggml_tensor * z = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3)), D, N);

    ggml_tensor * codes = nullptr;
    for (int64_t i0 = 0; i0 < N; i0 += APERTUS_VQ_CODES_CHUNK) {
        const int64_t n = std::min(APERTUS_VQ_CODES_CHUNK, N - i0);
        ggml_tensor * zc = ggml_view_2d(ctx0, z, D, n, z->nb[1], i0 * z->nb[1]);
        ggml_tensor * logits = ggml_mul_mat(ctx0, vq.codebook, zc); // [codebook_size, n]
        ggml_tensor * c = ggml_argmax(ctx0, logits);
        codes = codes ? ggml_concat(ctx0, codes, c, 0) : c;
    }

    ggml_set_name(codes, "out_codes");
    ggml_set_output(codes);
    ggml_build_forward_expand(gf, codes);

    return gf;
}
