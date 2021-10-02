#include <cstdio>
#include <hls_stream.h>

#include "ap_int.h"
#include "assert.h"
#include "defs.h"

ap_uint<COEF_WIDTH> mod_mult(ap_uint<COEF_WIDTH> x, ap_uint<COEF_WIDTH> y,
                             const ap_uint<COEF_WIDTH> q,
                             const ap_uint<COEF_WIDTH> q_inv) {
  ap_uint<2 * COEF_WIDTH> z;
  ap_uint<COEF_WIDTH> out;
  ap_uint<2 * COEF_WIDTH> t;

  z = x * y;
  t = z << COEF_WIDTH;
  t = (t * q_inv) >> COEF_WIDTH;
  t = t * q;
  out = z - t;
  if (out >= q)
    out -= q;
  return out;
}

void pe_proc(
  Coef_Bundle poly_in_act[N / COEF_PER_BEAT],
  Coef_Bundle poly_out_act[N / COEF_PER_BEAT],
  ap_uint<PARAM_WIDTH> weight_val) {
#pragma HLS inline
  for (unsigned int m = 0; m < N / COEF_PER_BEAT; m++) {
    Coef_Bundle in_bundle = poly_in_act[m];
    Coef_Bundle out_bundle = poly_out_act[m];
    for (unsigned i = 0; i < COEF_PER_BEAT; i++) {
#pragma HLS unroll
      out_bundle.data[i] += mod_mult(weight_val, in_bundle.data[i], q_0, q_0_inv);
    }
  }
}

static void comp(
    Coef_Bundle act_buffer[NUM_CU][N / COEF_PER_BEAT],
    Coef_Bundle partial_sum_buffer[NUM_CU][N / COEF_PER_BEAT],
    ap_uint<PARAM_WIDTH> weight_val_buffer[NUM_CU][MAX_COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_idx_buffer[NUM_CU][MAX_COUT_PER_CU * MAX_ROWS],
    unsigned k) {
  static ap_uint<8> current_ptr[NUM_CU];
  if (k == 1) {
    // reset arbitration
    for (unsigned int i = 0; i < NUM_CU; i++) {
#pragma HLS unroll
      current_ptr[i] = weight_idx_buffer[i][0];
    }
  }

  unsigned int in_start = k * NUM_CU;
  unsigned int in_end = (k + 1) * NUM_CU;

  bool bank_busy[NUM_CU];
  bool pe_busy[NUM_CU];
  int8_t assigned_pe_bank[NUM_CU];

  bool done = false;

  while (!done) {
    // arbitration
    for (unsigned int i = 0; i < NUM_CU; i++) {
#pragma HLS unroll
      bank_busy[i] = false;
      pe_busy[i] = false;
      assigned_pe_bank[i] = -1;

      ap_uint<8> bank_id = weight_idx_buffer[i][current_ptr[i]];
      if (bank_busy[bank_id % NUM_CU] == false) {
        assigned_pe_bank[i] = bank_id;
        bank_busy[bank_id] = true;
        pe_busy[i] = true;
      }
    }

    // computation
    for (unsigned int i = 0; i < NUM_CU; i++) {
#pragma HLS unroll factor=16
      if (pe_busy[i]) {
        pe_proc(act_buffer[assigned_pe_bank[i]], partial_sum_buffer[i],
            weight_val_buffer[i][current_ptr[i]]);
      }
    }

    // check_done
    done = true;
    for (unsigned int i = 0; i < NUM_CU; i++) {
#pragma HLS unroll
      if (pe_busy[i]) current_ptr[i]++;
      if (weight_idx_buffer[i][current_ptr[i]] < in_end) {
        done = false;
      }
    }
  }
}


void load_act(
    Coef_Bundle in_act_0[CIN_PER_BANK * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle in_act_1[CIN_PER_BANK * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle in_act_2[CIN_PER_BANK * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle in_act_3[CIN_PER_BANK * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle act_buffer[NUM_CU][N / COEF_PER_BEAT],
    unsigned int k, unsigned int j) {
      for (unsigned int m = 0; m < N / COEF_PER_BEAT; m++) {
        for (unsigned int i = 0; i < NUM_CU; i++) {
#pragma HLS unroll
          unsigned int per_bank_cu_offset = i % NUM_CU_PER_BANK;
          unsigned int polynomial_base_offset =
            per_bank_cu_offset * MAX_ACT_ITRS * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT;
          unsigned int polynomial_offset =
            polynomial_base_offset + (j % (R * NUM_CIPHERTEXT_POLY) * k) * N / COEF_PER_BEAT + m;
          if (i < 1 * NUM_CU_PER_BANK) {
            act_buffer[i][m] = in_act_0[per_bank_cu_offset];
          } else if (i < 2 * NUM_CU_PER_BANK) {
            act_buffer[i][m] = in_act_1[per_bank_cu_offset];
          } else if (i < 3 * NUM_CU_PER_BANK) {
            act_buffer[i][m] = in_act_2[per_bank_cu_offset];
          } else {
            act_buffer[i][m] = in_act_3[per_bank_cu_offset];
          }
        }
      }
}

void store_act(
  hls::stream<Coef_Bundle> pre_act_stream_0[NUM_CU_PER_BANK],
  hls::stream<Coef_Bundle> pre_act_stream_1[NUM_CU_PER_BANK],
  hls::stream<Coef_Bundle> pre_act_stream_2[NUM_CU_PER_BANK],
  hls::stream<Coef_Bundle> pre_act_stream_3[NUM_CU_PER_BANK],
  Coef_Bundle partial_sum_buffer[NUM_CU][N / COEF_PER_BEAT]) {
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

static void compute_linear(
    Coef_Bundle in_act_0[CIN_PER_BANK * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle in_act_1[CIN_PER_BANK * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle in_act_2[CIN_PER_BANK * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle in_act_3[CIN_PER_BANK * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    hls::stream<Coef_Bundle> pre_act_stream_0[NUM_CU_PER_BANK],
    hls::stream<Coef_Bundle> pre_act_stream_1[NUM_CU_PER_BANK],
    hls::stream<Coef_Bundle> pre_act_stream_2[NUM_CU_PER_BANK],
    hls::stream<Coef_Bundle> pre_act_stream_3[NUM_CU_PER_BANK],
    ap_uint<PARAM_WIDTH> weight_values_0[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_1[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_2[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_3[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_0[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_1[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_2[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_3[COUT_PER_BANK * MAX_ROWS]) {
  // activation buffer
  Coef_Bundle in_act_buffer_a[NUM_CU][N / COEF_PER_BEAT];
#pragma HLS array_partition variable=in_act_buffer_a cyclic factor=16 dim=1
#pragma HLS bind_storage variable=in_act_buffer_a type=RAM_S2P impl=URAM
  Coef_Bundle in_act_buffer_b[NUM_CU][N / COEF_PER_BEAT];
#pragma HLS array_partition variable=in_act_buffer_b cyclic factor=16 dim=1
#pragma HLS bind_storage variable=in_act_buffer_b type=RAM_S2P impl=URAM

  // partial sum buffer
  Coef_Bundle partial_sum_buffer_a[NUM_CU][N / COEF_PER_BEAT];
#pragma HLS array_partition variable=partial_sum_buffer_a cyclic factor=16 dim=1
#pragma HLS bind_storage variable=partial_sum_buffer_a type=RAM_S2P impl=URAM
  Coef_Bundle partial_sum_buffer_b[NUM_CU][N / COEF_PER_BEAT];
#pragma HLS array_partition variable=partial_sum_buffer_b cyclic factor=16 dim=1
#pragma HLS bind_storage variable=partial_sum_buffer_b type=RAM_S2P impl=URAM

  ap_uint<PARAM_WIDTH> weight_val_buffer[NUM_CU][MAX_COUT_PER_CU * MAX_ROWS];
#pragma HLS array_partition variable=weight_val_buffer cyclic factor=16 dim=1
#pragma HLS bind_storage variable=weight_val_buffer type=RAM_S2P impl=BRAM

  ap_uint<PARAM_WIDTH> weight_idx_buffer[NUM_CU][MAX_COUT_PER_CU * MAX_ROWS];
#pragma HLS array_partition variable=weight_idx_buffer cyclic factor=16 dim=1
#pragma HLS bind_storage variable=weight_idx_buffer type=RAM_S2P impl=BRAM

  // preload all the weights and the indices
  for (unsigned int j = 0; j < COUT_PER_CU; j++) {
    for (unsigned int k = 0; k < MAX_ROWS; k++) {
      for (unsigned int i = 0; i < NUM_CU; i++) {
        // TODO: Add NNZ support.
#pragma HLS unroll
        unsigned int per_bank_cu_offset = i % NUM_CU_PER_BANK;
        unsigned int weight_offset = j * NUM_CU_PER_BANK * MAX_ROWS +
                                     per_bank_cu_offset * MAX_ROWS + k;
        if (i < 1 * NUM_CU_PER_BANK) {
          weight_val_buffer[i][j * MAX_ROWS + k] = weight_values_0[weight_offset];
          weight_idx_buffer[i][j * MAX_ROWS + k] = weight_indices_0[weight_offset];
        } else if (i < 2 * NUM_CU_PER_BANK) {
          weight_val_buffer[i][j * MAX_ROWS + k] = weight_values_1[weight_offset];
          weight_idx_buffer[i][j * MAX_ROWS + k] = weight_indices_1[weight_offset];
        } else if (i < 3 * NUM_CU_PER_BANK) {
          weight_val_buffer[i][j * MAX_ROWS + k] = weight_values_2[weight_offset];
          weight_idx_buffer[i][j * MAX_ROWS + k] = weight_indices_2[weight_offset];
        } else {
          weight_val_buffer[i][j * MAX_ROWS + k] = weight_values_3[weight_offset];
          weight_idx_buffer[i][j * MAX_ROWS + k] = weight_indices_3[weight_offset];
        }
      }
    }
  }

  // First stream cout dimension then stream crt dimension.
  for (unsigned int i = 0; i < R * NUM_CIPHERTEXT_POLY; i++) {
    for (unsigned int j = 0; j < COUT_PER_CU; j++) {
      // prologue: load activation buffer
      load_act(in_act_0, in_act_1, in_act_2, in_act_3, in_act_buffer_a,
          /*k=*/0, /*j=*/j);

      // double buffer load / compute / store
      for (unsigned int k = 1; k < MAX_ACT_ITRS; k++) {
        if (k % 2 == 1) {
          load_act(in_act_0, in_act_1, in_act_2, in_act_3, in_act_buffer_b,
              /*k=*/k, /*j=*/j);

          if (j % 2 == 0) {
            comp(in_act_buffer_a, partial_sum_buffer_a, weight_val_buffer,
                weight_idx_buffer, k - 1);
          } else {
            comp(in_act_buffer_a, partial_sum_buffer_b, weight_val_buffer,
                weight_idx_buffer, k - 1);
          }

        } else {
          load_act(in_act_0, in_act_1, in_act_2, in_act_3, in_act_buffer_a,
              /*k=*/k, /*j=*/j);

          if (j % 2 == 0) {
            comp(in_act_buffer_b, partial_sum_buffer_a, weight_val_buffer,
                weight_idx_buffer, k - 1);
          } else {
            comp(in_act_buffer_b, partial_sum_buffer_b, weight_val_buffer,
                weight_idx_buffer, k - 1);
          }

        }
      }

      // Epilogue: finish the compute of the last compute iteration.
      if (MAX_ACT_ITRS % 2 == 1) {
        if (j % 2 == 0) {
          comp(in_act_buffer_a, partial_sum_buffer_a, weight_val_buffer,
              weight_idx_buffer, MAX_ACT_ITRS - 1);
        } else {
          comp(in_act_buffer_a, partial_sum_buffer_b, weight_val_buffer,
              weight_idx_buffer, MAX_ACT_ITRS - 1);
        }
      } else {
        if (j % 2 == 0) {
          comp(in_act_buffer_b, partial_sum_buffer_a, weight_val_buffer,
              weight_idx_buffer, MAX_ACT_ITRS - 1);
        } else {
          comp(in_act_buffer_b, partial_sum_buffer_b, weight_val_buffer,
              weight_idx_buffer, MAX_ACT_ITRS - 1);
        }
      }

      if (j % 2 == 0) {
        store_act(pre_act_stream_0, pre_act_stream_1, pre_act_stream_2,
            pre_act_stream_3, partial_sum_buffer_a);
      } else {
        store_act(pre_act_stream_0, pre_act_stream_1, pre_act_stream_2,
            pre_act_stream_3, partial_sum_buffer_b);
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
  for (unsigned int j = 0;
       j < COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT; j++) {
    for (unsigned int i = 0; i < NUM_CU_PER_BANK; i++) {
#pragma HLS unroll
      unsigned int offset =
          i * COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT;
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
    Coef_Bundle
        out_act_0[COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    // memory bank 1
    ap_uint<PARAM_WIDTH> NNZ_1[COUT_PER_BANK],
    ap_uint<PARAM_WIDTH> weight_values_1[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_1[COUT_PER_BANK * MAX_ROWS],
    Coef_Bundle in_act_1[CIN_PER_BANK * K_H * K_W * R * NUM_CIPHERTEXT_POLY *
                         N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_1[COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    // memory bank 2
    ap_uint<PARAM_WIDTH> NNZ_2[COUT_PER_BANK],
    ap_uint<PARAM_WIDTH> weight_values_2[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_2[COUT_PER_BANK * MAX_ROWS],
    Coef_Bundle in_act_2[CIN_PER_BANK * K_H * K_W * R * NUM_CIPHERTEXT_POLY *
                         N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_2[COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    // memory bank 3
    ap_uint<PARAM_WIDTH> NNZ_3[COUT_PER_BANK],
    ap_uint<PARAM_WIDTH> weight_values_3[COUT_PER_BANK * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_3[COUT_PER_BANK * MAX_ROWS],
    Coef_Bundle in_act_3[CIN_PER_BANK * K_H * K_W * R * NUM_CIPHERTEXT_POLY *
                         N / COEF_PER_BEAT],
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

#pragma HLS INTERFACE m_axi port = out_act_0 bundle = gmem0
#pragma HLS INTERFACE m_axi port = out_act_1 bundle = gmem1
#pragma HLS INTERFACE m_axi port = out_act_2 bundle = gmem2
#pragma HLS INTERFACE m_axi port = out_act_3 bundle = gmem3

  static hls::stream<Coef_Bundle, 32> pre_act_stream[NUM_MEM_BANKS]
                                                    [NUM_CU_PER_BANK];

#pragma HLS dataflow
  compute_linear(in_act_0, in_act_1, in_act_2,
                 in_act_3, pre_act_stream[0], pre_act_stream[1],
                 pre_act_stream[2], pre_act_stream[3], weight_values_0,
                 weight_values_1, weight_values_2, weight_values_3,
                 weight_indices_0, weight_indices_1, weight_indices_2,
                 weight_indices_3);
  store_act(pre_act_stream[0], pre_act_stream[1], pre_act_stream[2],
            pre_act_stream[3], out_act_0, out_act_1, out_act_2, out_act_3);
}
}
