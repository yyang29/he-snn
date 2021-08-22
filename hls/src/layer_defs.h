#ifndef LAYER_DEFS_H
#define LAYER_DEFS_H

#define HW_IN 14
#define C_IN 5
#define HW_OUT 5
#define C_OUT 50
#define K_H 5
#define K_W 5
#define STRIDE_HW 2
#define PAD_LEFT 0
#define PAD_RIGHT 0
#define PAD_TOP 0
#define PAD_BOTTOM 0

#define MAX_ROWS K_H * K_W * C_IN

#define N 8192
#define R 4
#define NUM_CIPHERTEXT_POLY 2

#define CIPHERTEXT N * R * NUM_CIPHERTEXT_POLY

#endif  // LAYER_DEFS_H