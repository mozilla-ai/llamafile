#ifdef __aarch64__
#define llamafile_mixmul llamafile_mixmul_arm80
#include "tinyblas_cpu_mixmul.inc"

/**
 * Returns number of shared memory bytes llamafile_mixmul() needs.
 */
size_t llamafile_mixmul_needs(const ggml_tensor *weights, const ggml_tensor *thought,
                              const ggml_tensor *plan) {
    ggml_compute_params params{};
    params.wsize = 0x7ffff000;
    params.wdata = (void *)0x1000;
    MixMul mm{&params, weights, thought, plan, 0};
    if (mm.allocate_shared_memory())
        return mm.get_allocated_bytes();
    else
        return 0;
}

#endif // __aarch64__
