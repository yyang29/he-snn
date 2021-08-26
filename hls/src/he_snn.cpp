#include <cstdio>
#include <hls_stream.h>

#include "ap_int.h"
#include "assert.h"
#include "defs.h"

ap_uint<COEF_WIDTH> mod_mult(ap_uint<COEF_WIDTH> x, ap_uint<COEF_WIDTH> y,
                             const ap_uint<COEF_WIDTH> q,
                             const ap_uint<COEF_WIDTH> q_inv) {
  ap_uint<2 * COEF_WIDTH> mult;
  ap_uint<COEF_WIDTH> out;
  ap_uint<2 * COEF_WIDTH> tmp;

  mult = x * y;
  tmp = mult >> (COEF_WIDTH - 1);
  tmp = (tmp * q_inv) >> (COEF_WIDTH + 1);
  tmp = tmp * q;
  out = mult - tmp;
  if (out >= q)
    out -= q;
  return out;
}

static void
load_act(Coef_Bundle in_act_0[CIN_PER_BANK * K_H * K_W * R *
                              NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
         Coef_Bundle in_act_1[CIN_PER_BANK * K_H * K_W * R *
                              NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
         Coef_Bundle in_act_2[CIN_PER_BANK * K_H * K_W * R *
                              NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
         Coef_Bundle in_act_3[CIN_PER_BANK * K_H * K_W * R *
                              NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
         hls::stream<Coef_Bundle> in_act_stream_0[NUM_CU_PER_BANK],
         hls::stream<Coef_Bundle> in_act_stream_1[NUM_CU_PER_BANK],
         hls::stream<Coef_Bundle> in_act_stream_2[NUM_CU_PER_BANK],
         hls::stream<Coef_Bundle> in_act_stream_3[NUM_CU_PER_BANK]) {
#pragma HLS pipeline II = 1
  // TODO: Polynomials are organized in the following order:
  // R * NUM_CIPHERTEXT_POLY * CIN_PER_BANK * K_H * K_W
in_act_cin_rd:
  for (unsigned int i = 0; i < CIN_PER_BANK; i += NUM_CU_PER_BANK) {
    unsigned int c_in_per_bank_left = CIN_PER_BANK - i;
    unsigned int c_in_current_itr = c_in_per_bank_left > NUM_CU_PER_BANK
                                        ? NUM_CU_PER_BANK
                                        : c_in_per_bank_left;
    unsigned int active_mask = (1 << c_in_current_itr) - 1;
    for (unsigned int j = 0; j < NUM_CU_PER_BANK; j++) {
#pragma HLS unroll
      bool active = ((1 << j) & active_mask) >> j;
      if (active == true) {
        unsigned int polynomial_base_offset =
            (i + j) * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT;
      in_act_hwr2_rd:
        for (unsigned int k = 0; k < R * NUM_CIPHERTEXT_POLY * K_H * K_W; k++) {
          int polynomial_offset = polynomial_base_offset + k;
          for (unsigned int m = 0; m < N / COEF_PER_BEAT; m++) {
            in_act_stream_0[j] << in_act_0[polynomial_offset];
            in_act_stream_1[j] << in_act_1[polynomial_offset];
            in_act_stream_2[j] << in_act_2[polynomial_offset];
            in_act_stream_3[j] << in_act_3[polynomial_offset];
            polynomial_offset++;
          }
        }
      }
    }
  }
}

static void
load_act_sparse(Coef_Bundle in_act_0[CIN_PER_BANK * K_H * K_W * R *
                                     NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
                Coef_Bundle in_act_1[CIN_PER_BANK * K_H * K_W * R *
                                     NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
                Coef_Bundle in_act_2[CIN_PER_BANK * K_H * K_W * R *
                                     NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
                Coef_Bundle in_act_3[CIN_PER_BANK * K_H * K_W * R *
                                     NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
                hls::stream<Coef_Bundle> in_act_stream_0[NUM_CU_PER_BANK],
                hls::stream<Coef_Bundle> in_act_stream_1[NUM_CU_PER_BANK],
                hls::stream<Coef_Bundle> in_act_stream_2[NUM_CU_PER_BANK],
                hls::stream<Coef_Bundle> in_act_stream_3[NUM_CU_PER_BANK]) {
  // TODO: Polynomials are organized in the following order:
  // R * NUM_CIPHERTEXT_POLY * CIN_PER_BANK * K_H * K_W
in_act_cin_rd:
    // Each activation is loaded COUT_PER_CU times.
    for (unsigned int n = 0; n < COUT_PER_CU; n++) {
      for (unsigned int k = 0; k < R * NUM_CIPHERTEXT_POLY * MAX_ACT_ITRS;
           k++) {
        for (unsigned int m = 0; m < N / COEF_PER_BEAT; m++) {

  for (unsigned int j = 0; j < NUM_CU_PER_BANK; j++) {
#pragma HLS unroll
    unsigned int polynomial_base_offset =
        j * MAX_ACT_ITRS * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT;
        int polynomial_offset = polynomial_base_offset + k * N / COEF_PER_BEAT;
          in_act_stream_0[j] << in_act_0[polynomial_offset];
          in_act_stream_1[j] << in_act_1[polynomial_offset];
          in_act_stream_2[j] << in_act_2[polynomial_offset];
          in_act_stream_3[j] << in_act_3[polynomial_offset];
          polynomial_offset++;
  }
        }
      }
    }
}

static void
compute_linear(hls::stream<Coef_Bundle> in_act_stream_0[NUM_CU_PER_BANK],
               hls::stream<Coef_Bundle> in_act_stream_1[NUM_CU_PER_BANK],
               hls::stream<Coef_Bundle> in_act_stream_2[NUM_CU_PER_BANK],
               hls::stream<Coef_Bundle> in_act_stream_3[NUM_CU_PER_BANK],
               hls::stream<Coef_Bundle> pre_act_stream_0[NUM_CU_PER_BANK],
               hls::stream<Coef_Bundle> pre_act_stream_1[NUM_CU_PER_BANK],
               hls::stream<Coef_Bundle> pre_act_stream_2[NUM_CU_PER_BANK],
               hls::stream<Coef_Bundle> pre_act_stream_3[NUM_CU_PER_BANK]) {
  // activation buffer
  Coef_Bundle in_act_buffer[NUM_CU][N / COEF_PER_BEAT];
#pragma HLS array_partition variable=in_act_buffer cyclic factor=32 dim=1 
  Coef_Bundle partial_sum_buffer[NUM_CU][N / COEF_PER_BEAT];
#pragma HLS array_partition variable=partial_sum_buffer cyclic factor=32 dim=1

    for (unsigned int j = 0; j < COUT_PER_CU * R * NUM_CIPHERTEXT_POLY; j++) {

      // accumulation
      // TODO: Change this based on sparsity
      for (unsigned int k = 0; k < MAX_ACT_ITRS; k++) {
        for (unsigned int m = 0; m < N / COEF_PER_BEAT; m++) {

  for (unsigned int i = 0; i < NUM_CU; i++) {
#pragma HLS unroll
    unsigned int per_bank_cu_offset = i % NUM_CU_PER_BANK;
          if (i < 1 * NUM_CU_PER_BANK) {
            in_act_buffer[i][m] = in_act_stream_0[per_bank_cu_offset].read();
          } else if (i < 2 * NUM_CU_PER_BANK) {
            in_act_buffer[i][m] = in_act_stream_1[per_bank_cu_offset].read();
          } else if (i < 3 * NUM_CU_PER_BANK) {
            in_act_buffer[i][m] = in_act_stream_2[per_bank_cu_offset].read();
          } else {
            in_act_buffer[i][m] = in_act_stream_3[per_bank_cu_offset].read();
          }
  }
        }
        for (unsigned int m = 0; m < N; m++) {

  for (unsigned int i = 0; i < NUM_CU; i++) {
#pragma HLS unroll
          unsigned row = m / COEF_PER_BEAT;
          unsigned col = m % COEF_PER_BEAT;
          partial_sum_buffer[i][row].data[col] =
              mod_mult(partial_sum_buffer[i][row].data[col],
                       in_act_buffer[i][row].data[col], q_0, q_0_inv);
  }
        }
      }

      // send out partial sum
      for (unsigned int m = 0; m < N / COEF_PER_BEAT; m++) {

  for (unsigned int i = 0; i < NUM_CU; i++) {
#pragma HLS unroll
    unsigned int per_bank_cu_offset = i % NUM_CU_PER_BANK;
        if (i < 1 * NUM_CU_PER_BANK) {
          pre_act_stream_0[per_bank_cu_offset] << partial_sum_buffer[i][m];
        } else if (i < 2 * NUM_CU_PER_BANK) {
          pre_act_stream_1[per_bank_cu_offset] << partial_sum_buffer[i][m];
        } else if (i < 3 * NUM_CU_PER_BANK) {
          pre_act_stream_2[per_bank_cu_offset] << partial_sum_buffer[i][m];
        } else if (i < 4 * NUM_CU_PER_BANK) {
          pre_act_stream_3[per_bank_cu_offset] << partial_sum_buffer[i][m];
        }
  }
      }
    }
}

static void store_act(
    hls::stream<Coef_Bundle> pre_act_stream_0[NUM_CU_PER_BANK],
    hls::stream<Coef_Bundle> pre_act_stream_1[NUM_CU_PER_BANK],
    hls::stream<Coef_Bundle> pre_act_stream_2[NUM_CU_PER_BANK],
    hls::stream<Coef_Bundle> pre_act_stream_3[NUM_CU_PER_BANK],
    Coef_Bundle
        out_act_0[COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_1[COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_2[COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle out_act_3[COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT]) {
    for (unsigned int j = 0; j < COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT - 1; j++) {
#pragma HLS unroll
  for (unsigned int i = 0; i < NUM_CU_PER_BANK; i++){
    unsigned int offset = i * COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT;
      out_act_0[offset + j] = pre_act_stream_0[i].read();
      out_act_1[offset + j] = pre_act_stream_1[i].read();
      out_act_2[offset + j] = pre_act_stream_2[i].read();
      out_act_3[offset + j] = pre_act_stream_3[i].read();
  }
    }
}

extern "C" {

void he_snn(
    // memory bank 0
    ap_uint<PARAM_WIDTH> NNZ_0[COUT_PER_BANK],
    ap_uint<PARAM_WIDTH> weight_values_0[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_0[COUT_PER_BANK * MAX_ROWS],
    Coef_Bundle in_act_0[CIN_PER_BANK * K_H * K_W * R * NUM_CIPHERTEXT_POLY *
                         N / COEF_PER_BEAT],
    // ap_uint<COEF_WIDTH> tf_ntt_0[NUM_CU_PER_BANK * N],
    // ap_uint<COEF_WIDTH> tf_intt_0[NUM_CU_PER_BANK * N],
    Coef_Bundle
        out_act_0[COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    // memory bank 1
    ap_uint<PARAM_WIDTH> NNZ_1[COUT_PER_BANK],
    ap_uint<PARAM_WIDTH> weight_values_1[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_1[COUT_PER_BANK * MAX_ROWS],
    Coef_Bundle in_act_1[CIN_PER_BANK * K_H * K_W * R * NUM_CIPHERTEXT_POLY *
                         N / COEF_PER_BEAT],
    // ap_uint<COEF_WIDTH> tf_ntt_1[NUM_CU_PER_BANK * N],
    // ap_uint<COEF_WIDTH> tf_intt_1[NUM_CU_PER_BANK * N],
    Coef_Bundle
        out_act_1[COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    // memory bank 2
    ap_uint<PARAM_WIDTH> NNZ_2[COUT_PER_BANK],
    ap_uint<PARAM_WIDTH> weight_values_2[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_2[COUT_PER_BANK * MAX_ROWS],
    Coef_Bundle in_act_2[CIN_PER_BANK * K_H * K_W * R * NUM_CIPHERTEXT_POLY *
                         N / COEF_PER_BEAT],
    // ap_uint<COEF_WIDTH> tf_ntt_2[NUM_CU_PER_BANK * N],
    // ap_uint<COEF_WIDTH> tf_intt_2[NUM_CU_PER_BANK * N],
    Coef_Bundle
        out_act_2[COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    // memory bank 3
    ap_uint<PARAM_WIDTH> NNZ_3[COUT_PER_BANK],
    ap_uint<PARAM_WIDTH> weight_values_3[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_3[COUT_PER_BANK * MAX_ROWS],
    Coef_Bundle in_act_3[CIN_PER_BANK * K_H * K_W * R * NUM_CIPHERTEXT_POLY *
                         N / COEF_PER_BEAT],
    // ap_uint<COEF_WIDTH> tf_ntt_3[NUM_CU_PER_BANK * N],
    // ap_uint<COEF_WIDTH> tf_intt_3[NUM_CU_PER_BANK * N],
    Coef_Bundle out_act_3[COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT]) {
#pragma HLS INTERFACE m_axi port = NNZ_0 bundle = gmem0
#pragma HLS INTERFACE m_axi port = NNZ_1 bundle = gmem1
#pragma HLS INTERFACE m_axi port = NNZ_2 bundle = gmem2
#pragma HLS INTERFACE m_axi port = NNZ_3 bundle = gmem3

#pragma HLS INTERFACE m_axi port = weight_values_0 bundle = gmem0
#pragma HLS INTERFACE m_axi port = weight_values_1 bundle = gmem1
#pragma HLS INTERFACE m_axi port = weight_values_2 bundle = gmem2
#pragma HLS INTERFACE m_axi port = weight_values_3 bundle = gmem3

#pragma HLS INTERFACE m_axi port = weight_indices_0 bundle = gmem0
#pragma HLS INTERFACE m_axi port = weight_indices_1 bundle = gmem1
#pragma HLS INTERFACE m_axi port = weight_indices_2 bundle = gmem2
#pragma HLS INTERFACE m_axi port = weight_indices_3 bundle = gmem3

#pragma HLS INTERFACE m_axi port = in_act_0 bundle = gmem0
#pragma HLS INTERFACE m_axi port = in_act_1 bundle = gmem1
#pragma HLS INTERFACE m_axi port = in_act_2 bundle = gmem2
#pragma HLS INTERFACE m_axi port = in_act_3 bundle = gmem3

// #pragma HLS INTERFACE m_axi port = tf_ntt_0 bundle = gmem0
// #pragma HLS INTERFACE m_axi port = tf_ntt_1 bundle = gmem1
// #pragma HLS INTERFACE m_axi port = tf_ntt_2 bundle = gmem2
// #pragma HLS INTERFACE m_axi port = tf_ntt_3 bundle = gmem3
 
// #pragma HLS INTERFACE m_axi port = tf_intt_0 bundle = gmem0
// #pragma HLS INTERFACE m_axi port = tf_intt_1 bundle = gmem1
// #pragma HLS INTERFACE m_axi port = tf_intt_2 bundle = gmem2
// #pragma HLS INTERFACE m_axi port = tf_intt_3 bundle = gmem3

#pragma HLS INTERFACE m_axi port = out_act_0 bundle = gmem0
#pragma HLS INTERFACE m_axi port = out_act_1 bundle = gmem1
#pragma HLS INTERFACE m_axi port = out_act_2 bundle = gmem2
#pragma HLS INTERFACE m_axi port = out_act_3 bundle = gmem3

  static hls::stream<Coef_Bundle> in_act_stream[NUM_MEM_BANKS][NUM_CU_PER_BANK];
  static hls::stream<Coef_Bundle> pre_act_stream[NUM_MEM_BANKS]
                                                [NUM_CU_PER_BANK];

#pragma HLS dataflow
  load_act_sparse(in_act_0, in_act_1, in_act_2, in_act_3, in_act_stream[0],
                  in_act_stream[1], in_act_stream[2], in_act_stream[3]);
  compute_linear(in_act_stream[0], in_act_stream[1], in_act_stream[2],
                 in_act_stream[3], pre_act_stream[0], pre_act_stream[1],
                 pre_act_stream[2], pre_act_stream[3]);
  store_act(pre_act_stream[0], pre_act_stream[1], pre_act_stream[2],
            pre_act_stream[3], out_act_0, out_act_1, out_act_2, out_act_3);
}
}
