#include "models.h"

#include <cmath>
#include <cstring>
#include <vector>

// Apertus 1.5 audio tokenizer (WavTokenizer encoder), encode only: 24 kHz waveform -> one code per hop
// ref: transformers WavTokenizerEncoderModel.encode
// activations are [C, T]

static int64_t div_ceil(int64_t a, int64_t b) {
    return a >= 0 ? (a + b - 1) / b : -((-a) / b);
}

// non-causal conv with reflect padding, see WavTokenizerConv1d.forward
// x: [IC, T], w: [K, IC, OC] -> [OC, ceil(T / stride)]
ggml_tensor * clip_graph_apertus1p5a::conv1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride) const {
    const int64_t K    = w->ne[0];
    const int64_t Cin  = w->ne[1];
    const int64_t Cout = w->ne[2];
    const int64_t T    = x->ne[1];
    GGML_ASSERT(x->ne[0] == Cin);

    // trailing padding so that every input sample is covered: ceil(T / stride) outputs
    const int64_t p_total  = K - stride;
    const int64_t n_frames = div_ceil(T - K + p_total, stride);
    const int64_t p_extra  = n_frames * stride + K - p_total - T;
    const int64_t n_out    = n_frames + 1;
    const int64_t p_right = p_total / 2;
    const int64_t p_left  = p_total - p_right;

    if (p_left > 0 || p_right + p_extra > 0) {
        // reflect padding works on dim 0: pad [T, C], then go back to [C, T]
        ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));
        const int64_t max_pad = std::max(p_left, p_right + p_extra);
        int64_t zero_pad = 0;
        if (T <= max_pad) {
            // too short to reflect: extend with zeros first, then drop them again, like _pad1d()
            zero_pad = max_pad - T + 1;
            xt = ggml_pad_ext(ctx0, xt, 0, (int) zero_pad, 0, 0, 0, 0, 0, 0);
        }
        xt = ggml_pad_reflect_1d(ctx0, xt, (int) p_left, (int) (p_right + p_extra));
        if (zero_pad > 0) {
            xt = ggml_view_2d(ctx0, xt, xt->ne[0] - zero_pad, xt->ne[1], xt->nb[1], 0);
        }
        x = ggml_cont(ctx0, ggml_transpose(ctx0, xt));
    }
    GGML_ASSERT((x->ne[1] - K) / stride + 1 == n_out);

    // sum of one matmul per tap
    ggml_tensor * wt = K > 1 ? ggml_cont(ctx0, ggml_permute(ctx0, w, 2, 0, 1, 3)) : ggml_reshape_2d(ctx0, w, Cin, Cout); // [Cin, Cout, K]
    ggml_tensor * out = nullptr;
    for (int64_t k = 0; k < K; k++) {
        ggml_tensor * wk = K > 1 ? ggml_view_2d(ctx0, wt, Cin, Cout, wt->nb[1], k * wt->nb[2]) : wt;
        ggml_tensor * xk = ggml_view_2d(ctx0, x, Cin, n_out, stride * x->nb[1], k * x->nb[1]);
        ggml_tensor * yk = ggml_mul_mat(ctx0, wk, xk); // [Cout, n_out]
        out = out ? ggml_add(ctx0, out, yk) : yk;
    }
    if (b) {
        out = ggml_add(ctx0, out, b);
    }
    return out;
}

// one LSTM layer, sequential over time, on the CPU
// src[0]: input gates W_ih x_t + b_ih + b_hh [4H, T], src[1]: W_hh transposed [4H, H]
// gate order i, f, g, o as in torch.nn.LSTM, zero initial state
static void apertus_lstm_recurrence(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    GGML_UNUSED(userdata);
    if (ith != 0) {
        return;
    }
    const ggml_tensor * gx  = dst->src[0];
    const ggml_tensor * wht = dst->src[1];
    const int64_t H  = dst->ne[0];
    const int64_t T  = dst->ne[1];
    const int64_t H4 = 4 * H;
    GGML_ASSERT(gx->ne[0] == H4 && wht->ne[0] == H4 && wht->ne[1] == H);
    GGML_ASSERT(gx->type == GGML_TYPE_F32 && wht->type == GGML_TYPE_F32 && ggml_is_contiguous(wht));

    std::vector<float> h(H, 0.0f), c(H, 0.0f), g(H4);
    const float * W = (const float *) wht->data;
    auto sigmoid = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };

    for (int64_t t = 0; t < T; t++) {
        memcpy(g.data(), (const char *) gx->data + t * gx->nb[1], H4 * sizeof(float));
        // g += W_hh h, as one saxpy per column of W_hh
        for (int64_t k = 0; k < H; k++) {
            const float hk = h[k];
            const float * wk = W + k * H4;
            for (int64_t r = 0; r < H4; r++) {
                g[r] += wk[r] * hk;
            }
        }
        for (int64_t j = 0; j < H; j++) {
            const float ig = sigmoid(g[j]);
            const float fg = sigmoid(g[H + j]);
            const float cg = std::tanh(g[2 * H + j]);
            const float og = sigmoid(g[3 * H + j]);
            c[j] = fg * c[j] + ig * cg;
            h[j] = og * std::tanh(c[j]);
        }
        memcpy((char *) dst->data + t * dst->nb[1], h.data(), H * sizeof(float));
    }
}

ggml_cgraph * clip_graph_apertus1p5a::build() {
    const auto & codec = model.apertus_codec;

    ggml_tensor * inp = build_inp_raw(1);
    // [T, 1] -> [1, T]
    ggml_tensor * cur = ggml_reshape_2d(ctx0, inp, 1, inp->ne[0]);

    cur = conv1d(cur, codec.conv_in_w, codec.conv_in_b, 1);

    for (const auto & st : codec.stages) {
        ggml_tensor * h = conv1d(ggml_elu(ctx0, cur), st.conv1_w, st.conv1_b, 1);
        h = conv1d(ggml_elu(ctx0, h), st.conv2_w, st.conv2_b, 1);
        ggml_tensor * res = st.short_w ? conv1d(cur, st.short_w, st.short_b, 1) : cur;
        cur = ggml_add(ctx0, res, h);

        const int stride = (int) (st.down_w->ne[0] / 2);
        cur = conv1d(ggml_elu(ctx0, cur), st.down_w, st.down_b, stride);
        cb(cur, "stage_out", -1);
    }

    // LSTM with a residual connection
    ggml_tensor * lstm_in = cur;
    for (const auto & l : codec.lstm) {
        ggml_tensor * gx = ggml_mul_mat(ctx0, l.w_ih, cur);
        gx = ggml_add(ctx0, ggml_add(ctx0, gx, l.b_ih), l.b_hh);
        ggml_tensor * wht = ggml_cont(ctx0, ggml_transpose(ctx0, l.w_hh));
        ggml_tensor * args[] = { gx, wht };
        cur = ggml_custom_4d(ctx0, GGML_TYPE_F32, l.w_hh->ne[0], cur->ne[1], 1, 1, args, 2, apertus_lstm_recurrence, 1, nullptr);
    }
    cur = ggml_add(ctx0, cur, lstm_in);

    cur = conv1d(ggml_elu(ctx0, cur), codec.conv_out_w, codec.conv_out_b, 1);
    cb(cur, "latent", -1);

    // nearest code in euclidean distance: argmax of 2 x.e - |e|^2
    ggml_tensor * e2 = ggml_sum_rows(ctx0, ggml_sqr(ctx0, codec.codebook)); // [1, n_code]
    e2 = ggml_reshape_2d(ctx0, e2, e2->ne[1], 1);
    ggml_tensor * logits = ggml_mul_mat(ctx0, codec.codebook, cur); // [n_code, T]
    logits = ggml_sub(ctx0, ggml_scale(ctx0, logits, 2.0f), e2);
    ggml_tensor * codes = ggml_argmax(ctx0, logits);

    ggml_set_name(codes, "out_codes");
    ggml_set_output(codes);
    ggml_build_forward_expand(gf, codes);

    return gf;
}
