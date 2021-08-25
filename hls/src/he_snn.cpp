#include <cstdio>
#include <hls_stream.h>

#include "ap_int.h"
#include "assert.h"
#include "defs.h"

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

// static void
// compute_linear(hls::stream<Polynomial> in_act_stream_0[NUM_CU_PER_BANK],
//                hls::stream<Polynomial> in_act_stream_1[NUM_CU_PER_BANK],
//                hls::stream<Polynomial> in_act_stream_2[NUM_CU_PER_BANK],
//                hls::stream<Polynomial> in_act_stream_3[NUM_CU_PER_BANK],
//                hls::stream<Polynomial> pre_act_stream_0[NUM_CU_PER_BANK],
//                hls::stream<Polynomial> pre_act_stream_1[NUM_CU_PER_BANK],
//                hls::stream<Polynomial> pre_act_stream_2[NUM_CU_PER_BANK],
//                hls::stream<Polynomial> pre_act_stream_3[NUM_CU_PER_BANK]) {
// #pragma HLS pipeline II = 1
//   // activation buffer
//   Polynomial in_act_buffer[NUM_CU];
// #pragma HLS array_partition variable = in_act_buffer type = cyclic factor = \
//     NUM_CU dim = 1
//   Polynomial partial_sum_buffer[NUM_CU];
// #pragma HLS array_partition variable = partial_sum_buffer type = \
//     cyclic factor = NUM_CU dim = 1

//   for (unsigned int i = 0; i < NUM_CU; i++) {
// #pragma HLS unroll
//     unsigned int per_bank_cu_offset = i % NUM_CU_PER_BANK;
//     for (unsigned int j = 0; j < R * NUM_CIPHERTEXT_POLY; j++) {
//       // accumulation
//       // TODO: Change this based on sparsity
//       for (unsigned int k = 0; k < K_H * K_W; k++) {
//         if (i < 1 * NUM_CU_PER_BANK) {
//           in_act_buffer[i] = in_act_stream_0[per_bank_cu_offset].read();
//         } else if (i < 2 * NUM_CU_PER_BANK) {
//           in_act_buffer[i] = in_act_stream_1[per_bank_cu_offset].read();
//         } else if (i < 3 * NUM_CU_PER_BANK) {
//           in_act_buffer[i] = in_act_stream_2[per_bank_cu_offset].read();
//         } else {
//           in_act_buffer[i] = in_act_stream_3[per_bank_cu_offset].read();
//         }
//         for (unsigned int m = 0; m < N; m++) {
//           partial_sum_buffer[i].data[m] += in_act_buffer[i].data[m];
//           partial_sum_buffer[i].data[m] %= q_0;
//         }
//       }
//       // send out partial sum
//       if (i < 1 * NUM_CU_PER_BANK) {
//         pre_act_stream_0[per_bank_cu_offset] << partial_sum_buffer[i];
//       } else if (i < 2 * NUM_CU_PER_BANK) {
//         pre_act_stream_1[per_bank_cu_offset] << partial_sum_buffer[i];
//       } else if (i < 3 * NUM_CU_PER_BANK) {
//         pre_act_stream_2[per_bank_cu_offset] << partial_sum_buffer[i];
//       } else if (i < 4 * NUM_CU_PER_BANK) {
//         pre_act_stream_3[per_bank_cu_offset] << partial_sum_buffer[i];
//       }
//     }
//   }
// }

// static void store_act(hls::vector<unsigned int, 16> *out,
//                          hls::stream<hls::vector<unsigned int, 16>>
//                          &out_stream, int vSize) {
// mem_wr:
//   for (int i = 0; i < vSize; i++) {
// #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
//     out[i] = out_stream.read();
//   }
// }

extern "C" {

void he_snn(
    // memory bank 0
    ap_uint<PARAM_WIDTH> NNZ_0[COUT_PER_BANK],
    ap_uint<PARAM_WIDTH> weight_values_0[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_0[COUT_PER_BANK * MAX_ROWS],
    Coef_Bundle in_act_0[CIN_PER_BANK * K_H * K_W * R * NUM_CIPHERTEXT_POLY *
                         N / COEF_PER_BEAT],
    ap_uint<COEF_WIDTH> tf_ntt_0[NUM_CU_PER_BANK * N],
    ap_uint<COEF_WIDTH> tf_intt_0[NUM_CU_PER_BANK * N],
    Coef_Bundle
        out_act_0[COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    // memory bank 1
    ap_uint<PARAM_WIDTH> NNZ_1[COUT_PER_BANK],
    ap_uint<PARAM_WIDTH> weight_values_1[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_1[COUT_PER_BANK * MAX_ROWS],
    Coef_Bundle in_act_1[CIN_PER_BANK * K_H * K_W * R * NUM_CIPHERTEXT_POLY *
                         N / COEF_PER_BEAT],
    ap_uint<COEF_WIDTH> tf_ntt_1[NUM_CU_PER_BANK * N],
    ap_uint<COEF_WIDTH> tf_intt_1[NUM_CU_PER_BANK * N],
    Coef_Bundle
        out_act_1[COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    // memory bank 2
    ap_uint<PARAM_WIDTH> NNZ_2[COUT_PER_BANK],
    ap_uint<PARAM_WIDTH> weight_values_2[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_2[COUT_PER_BANK * MAX_ROWS],
    Coef_Bundle in_act_2[CIN_PER_BANK * K_H * K_W * R * NUM_CIPHERTEXT_POLY *
                         N / COEF_PER_BEAT],
    ap_uint<COEF_WIDTH> tf_ntt_2[NUM_CU_PER_BANK * N],
    ap_uint<COEF_WIDTH> tf_intt_2[NUM_CU_PER_BANK * N],
    Coef_Bundle
        out_act_2[COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    // memory bank 3
    ap_uint<PARAM_WIDTH> NNZ_3[COUT_PER_BANK],
    ap_uint<PARAM_WIDTH> weight_values_3[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_3[COUT_PER_BANK * MAX_ROWS],
    Coef_Bundle in_act_3[CIN_PER_BANK * K_H * K_W * R * NUM_CIPHERTEXT_POLY *
                         N / COEF_PER_BEAT],
    ap_uint<COEF_WIDTH> tf_ntt_3[NUM_CU_PER_BANK * N],
    ap_uint<COEF_WIDTH> tf_intt_3[NUM_CU_PER_BANK * N],
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

#pragma HLS INTERFACE m_axi port = tf_ntt_0 bundle = gmem0
#pragma HLS INTERFACE m_axi port = tf_ntt_1 bundle = gmem1
#pragma HLS INTERFACE m_axi port = tf_ntt_2 bundle = gmem2
#pragma HLS INTERFACE m_axi port = tf_ntt_3 bundle = gmem3

#pragma HLS INTERFACE m_axi port = tf_intt_0 bundle = gmem0
#pragma HLS INTERFACE m_axi port = tf_intt_1 bundle = gmem1
#pragma HLS INTERFACE m_axi port = tf_intt_2 bundle = gmem2
#pragma HLS INTERFACE m_axi port = tf_intt_3 bundle = gmem3

#pragma HLS INTERFACE m_axi port = out_act_0 bundle = gmem0
#pragma HLS INTERFACE m_axi port = out_act_1 bundle = gmem1
#pragma HLS INTERFACE m_axi port = out_act_2 bundle = gmem2
#pragma HLS INTERFACE m_axi port = out_act_3 bundle = gmem3

  static hls::stream<Coef_Bundle, 1152> in_act_stream[NUM_MEM_BANKS][NUM_CU_PER_BANK];
  static hls::stream<Coef_Bundle, 1152> pre_act_stream[NUM_MEM_BANKS]
                                                [NUM_CU_PER_BANK];
  static hls::stream<Coef_Bundle, 1152> out_act_stream[NUM_MEM_BANKS]
                                                [NUM_CU_PER_BANK];

#pragma HLS dataflow
  load_act(in_act_0, in_act_1, in_act_2, in_act_3, in_act_stream[0],
           in_act_stream[1], in_act_stream[2], in_act_stream[3]);
  // compute_linear();
  // compute_non_linear();
  // store_act();

  out_act_0[0] = in_act_stream[0][0].read();
  out_act_1[0] = in_act_stream[1][0].read();
  out_act_2[0] = in_act_stream[2][0].read();
  out_act_3[0] = in_act_stream[3][0].read();
}
}