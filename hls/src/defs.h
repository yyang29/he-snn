#include <cmath>

#include <ap_int.h>

#ifndef LAYER_DEFS_H
#define LAYER_DEFS_H

// per layer config
#define HW_IN 14
#define C_IN 5
#define HW_OUT 5
// TODO: Fix C_OUT issue when it is not a multiple of 32.
#define C_OUT 64
#define K_H 5
#define K_W 5
#define STRIDE_HW 2
#define PAD_LEFT 0
#define PAD_RIGHT 0
#define PAD_TOP 0
#define PAD_BOTTOM 0
#define MAX_ACT_ITRS 1

// hardware config
#define NUM_CU 32
#define NUM_MEM_BANKS 4
#define NUM_CU_PER_BANK (NUM_CU / NUM_MEM_BANKS)

// derived config
#define MAX_ROWS (K_H * K_W * C_IN)

#define N 8192
#define R 4
#define NUM_CIPHERTEXT_POLY 2

#define CIPHERTEXT (N * R * NUM_CIPHERTEXT_POLY)

#define BYTES_INT16 2
#define BYTES_INT64 8

#define PARAM_WIDTH 16
#define COEF_WIDTH 64

const int COUT_PER_BANK = ceil((float)C_OUT / (float)NUM_MEM_BANKS);
const int CIN_PER_BANK = ceil((float)C_IN / (float)NUM_MEM_BANKS);
const int COUT_PER_CU = ceil((float)COUT_PER_BANK / (float)NUM_CU_PER_BANK);
const int CIN_PER_CU = ceil((float)CIN_PER_BANK / (float)NUM_CU_PER_BANK);

struct Polynomial {
  ap_uint<COEF_WIDTH> data[N];
};

#define COEF_BUNDLE_BITS 512
#define COEF_PER_BEAT (COEF_BUNDLE_BITS / COEF_WIDTH)

struct Coef_Bundle {
  ap_uint<COEF_WIDTH> data[COEF_PER_BEAT];
};

const ap_uint<COEF_WIDTH> q_0=184467440737095;
const ap_uint<COEF_WIDTH> q_0_inv=184467440737095;

#endif  // LAYER_DEFS_H