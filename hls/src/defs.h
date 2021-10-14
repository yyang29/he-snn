#include <ap_int.h>

#ifndef LAYER_DEFS_H
#define LAYER_DEFS_H

// per layer config
#define HW_IN 16
#define C_IN 32
#define HW_OUT 16
#define C_OUT 64
#define COUT_PER_BANK 16
#define COUT_PER_CU 4
#define CIN_PER_CU 4
#define K_H 3
#define K_W 3
#define STRIDE_HW 2
#define PAD_LEFT 0
#define PAD_RIGHT 0
#define PAD_TOP 0
#define PAD_BOTTOM 0

// hardware config
#define NUM_CU 8
#define NUM_MEM_BANKS 4
#define NUM_CU_PER_BANK (NUM_CU / NUM_MEM_BANKS)

// derived config based on cifar
#define MAX_ACT_ITRS 512
#define ON_CHIP_W_MAX_ROWS 4096
#define OFF_CHIP_W_MAX_ROWS 8192

#define N 8192
#define R 7
#define NUM_CIPHERTEXT_POLY 2

#define CIPHERTEXT (N * R * NUM_CIPHERTEXT_POLY)

#define BYTES_INT64 8

#define PARAM_WIDTH 64
#define COEF_WIDTH 64

struct Polynomial {
  ap_uint<COEF_WIDTH> data[N];
};

#define COEF_BUNDLE_BITS 1024
#define COEF_PER_BEAT (COEF_BUNDLE_BITS / COEF_WIDTH)

// struct Coef_Bundle {
//   ap_uint<COEF_WIDTH> data[COEF_PER_BEAT];
// };

typedef ap_uint<COEF_BUNDLE_BITS> Coef_Bundle;

const ap_uint<COEF_WIDTH> q_0[7] = {
  184467440727095, 184542340737095, 184467440737095, 184467440373095,
  184461666667095, 184423333737095, 184458580737095
};

const ap_uint<COEF_WIDTH> q_0_inv[7] = {
  123467835627095, 123579430737095, 121357440737095, 184467440373095,
  184459595997095, 184421324568095, 184458235737095
};

#endif  // LAYER_DEFS_H
