// q8_0_layout_test: Diagnostic for block_q8_0 vs block_q8_0_x4 memory layout
//
// This is a standalone diagnostic test, NOT part of the regular test suite.
// It documents why the IQK code in iqk_mul_mat.inc gathers scale values
// individually rather than casting block_q8_0* to block_q8_0_x4*.
//
// The issue: block_q8_0_x4 expects a packed layout [d0,d1,d2,d3,qs0,qs1,qs2,qs3]
// but actual block_q8_0 arrays have interleaved layout [d0,qs0,d1,qs1,d2,qs2,d3,qs3].
// Casting would read qs bytes as scale values, producing garbage/inf/nan.
//
// See commit 474c8b6 for the related fix to iqk_mul_mat.inc.
//
// To build: make o/$(MODE)/tests/sgemm/q8_0_layout_test
// To run:   ./o//tests/sgemm/q8_0_layout_test

#include <cstdio>
#include <cstdint>
#include <cstring>

// Mimic ggml types
typedef uint16_t ggml_half;
#define QK8_0 32

// Standard block_q8_0 (from ggml-common.h)
typedef struct {
    ggml_half d;       // delta (scale)
    int8_t  qs[QK8_0]; // quants
} block_q8_0;

// Packed block_q8_0_x4 (from iqk_mul_mat.inc)
typedef struct {
    ggml_half d[4];
    int8_t qs[4*QK8_0];
} block_q8_0_x4;

int main() {
    printf("=== block_q8_0 vs block_q8_0_x4 Layout Test ===\n\n");

    printf("Sizes:\n");
    printf("  sizeof(block_q8_0)    = %zu bytes\n", sizeof(block_q8_0));
    printf("  sizeof(block_q8_0_x4) = %zu bytes\n", sizeof(block_q8_0_x4));
    printf("  4 * sizeof(block_q8_0) = %zu bytes\n", 4 * sizeof(block_q8_0));
    printf("\n");

    // Create 4 block_q8_0 blocks with known values
    block_q8_0 blocks[4];
    for (int i = 0; i < 4; i++) {
        // Set scale to a distinctive value (as fp16 bits)
        // Use simple values: 0x1000, 0x2000, 0x3000, 0x4000
        blocks[i].d = 0x1000 * (i + 1);
        // Fill qs with block index for easy identification
        for (int j = 0; j < QK8_0; j++) {
            blocks[i].qs[j] = (int8_t)(i * 10 + j);
        }
    }

    printf("Created 4 block_q8_0 blocks:\n");
    for (int i = 0; i < 4; i++) {
        printf("  blocks[%d].d = 0x%04x, qs[0..2] = %d,%d,%d\n",
               i, blocks[i].d, blocks[i].qs[0], blocks[i].qs[1], blocks[i].qs[2]);
    }
    printf("\n");

    // Cast to block_q8_0_x4 (this is what IQK does)
    const block_q8_0_x4 *x4 = (const block_q8_0_x4 *)blocks;

    printf("Reinterpreted as block_q8_0_x4:\n");
    printf("  x4->d[0] = 0x%04x (expected 0x1000)\n", x4->d[0]);
    printf("  x4->d[1] = 0x%04x (expected 0x2000)\n", x4->d[1]);
    printf("  x4->d[2] = 0x%04x (expected 0x3000)\n", x4->d[2]);
    printf("  x4->d[3] = 0x%04x (expected 0x4000)\n", x4->d[3]);
    printf("\n");

    // Check what the d values actually are
    printf("Memory layout analysis:\n");
    uint8_t *raw = (uint8_t *)blocks;
    printf("  Bytes at offset 0-1 (block 0 d): 0x%02x%02x\n", raw[1], raw[0]);
    printf("  Bytes at offset 2-3 (block 0 qs[0-1]): %d, %d\n", (int8_t)raw[2], (int8_t)raw[3]);
    printf("  Bytes at offset %zu-%zu (block 1 d): 0x%02x%02x\n",
           sizeof(block_q8_0), sizeof(block_q8_0)+1,
           raw[sizeof(block_q8_0)+1], raw[sizeof(block_q8_0)]);
    printf("\n");

    // What IQK's load_scales() actually reads when it does vld1_f16(x4->d)
    printf("What IQK reads as 4 fp16 scale values:\n");
    printf("  Value 0: 0x%04x (bytes at offset 0-1)\n", x4->d[0]);
    printf("  Value 1: 0x%04x (bytes at offset 2-3)\n", x4->d[1]);
    printf("  Value 2: 0x%04x (bytes at offset 4-5)\n", x4->d[2]);
    printf("  Value 3: 0x%04x (bytes at offset 6-7)\n", x4->d[3]);
    printf("\n");

    // Compare with what it SHOULD read
    printf("What IQK SHOULD read:\n");
    printf("  Value 0: 0x%04x (blocks[0].d)\n", blocks[0].d);
    printf("  Value 1: 0x%04x (blocks[1].d)\n", blocks[1].d);
    printf("  Value 2: 0x%04x (blocks[2].d)\n", blocks[2].d);
    printf("  Value 3: 0x%04x (blocks[3].d)\n", blocks[3].d);
    printf("\n");

    // Check if there's a mismatch
    bool mismatch = false;
    for (int i = 0; i < 4; i++) {
        if (x4->d[i] != blocks[i].d) {
            mismatch = true;
            break;
        }
    }

    if (mismatch) {
        printf("*** MISMATCH DETECTED ***\n");
        printf("The block_q8_0_x4 cast reads WRONG values!\n");
        printf("x4->d[1] reads bytes 2-3 which are actually blocks[0].qs[0-1] = %d, %d\n",
               blocks[0].qs[0], blocks[0].qs[1]);
        printf("These int8 values interpreted as fp16 will produce garbage/inf/nan!\n");
    } else {
        printf("No mismatch - layouts are compatible.\n");
    }

    return mismatch ? 1 : 0;
}
