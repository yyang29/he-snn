#include <hls_stream.h>
#include <iostream>

#include "ap_int.h"
#include "assert.h"
#include "defs.h"

ap_uint<COEF_WIDTH> mod_add(ap_uint<COEF_WIDTH> x, ap_uint<COEF_WIDTH> y,
                            const ap_uint<COEF_WIDTH> q) {
#pragma hls inline
  ap_uint<2 * COEF_WIDTH> z;

  z = x + y;
  if (z >= q)
    z -= q;
  // std::cout << "Computation x: " << x << " y: " << y
  //           << " out: " << out << std::endl;
  return z;
}

ap_uint<COEF_WIDTH> mod_mult(ap_uint<COEF_WIDTH> x, ap_uint<COEF_WIDTH> y,
                             const ap_uint<COEF_WIDTH> q,
                             const ap_uint<COEF_WIDTH> q_inv) {
#pragma hls inline
  ap_uint<2 * COEF_WIDTH> z;
  ap_uint<COEF_WIDTH> out;
  ap_uint<2 * COEF_WIDTH> t;

  z = x * y;
  t = z >> COEF_WIDTH;
  t = (t * q_inv) >> COEF_WIDTH;
  t = t * q;
  out = z - t;
  if (out >= q)
    out -= q;
  // std::cout << "Computation x: " << x << " y: " << y
  //           << " out: " << out << std::endl;
  return out;
}

// void pe_proc(Coef_Bundle poly_in_act[N / COEF_PER_BEAT],
//              Coef_Bundle poly_out_act[N / COEF_PER_BEAT],
//              ap_uint<PARAM_WIDTH> weight_val, unsigned int rns) {
// #pragma hls inline
//   for (unsigned int m = 0; m < N / COEF_PER_BEAT; m++) {
//     Coef_Bundle in_bundle = poly_in_act[m];
//     Coef_Bundle out_bundle = poly_out_act[m];
//     for (unsigned i = 0; i < COEF_PER_BEAT; i++) {
// #pragma HLS unroll
//       out_bundle.data[i] +=
//           mod_mult(weight_val, in_bundle.data[i], q_0[rns], q_0_inv[rns]);
//     }
//     poly_out_act[m] = out_bundle;
//   }
// }

void comp(Coef_Bundle act_buffer[NUM_CU][N / COEF_PER_BEAT],
          Coef_Bundle partial_sum_buffer[NUM_CU][N / COEF_PER_BEAT],
          ap_uint<PARAM_WIDTH> weight_val_buffer[NUM_CU][ON_CHIP_W_MAX_ROWS],
          ap_uint<PARAM_WIDTH> weight_idx_buffer[NUM_CU][ON_CHIP_W_MAX_ROWS],
          ap_uint<PARAM_WIDTH> in_itr_count[MAX_ACT_ITRS], unsigned int k,
          unsigned int j, unsigned int rns) {

  static unsigned int counter;
  // reset counter when a new rns component starts.
  if (j == 0) {
    counter = 0;
  }

  std::cout << "compute " << k << " iteration count: " << in_itr_count[k]
            << " RNS term: " << rns << std::endl;

act_loop:
  for (unsigned int itr = 0; itr < in_itr_count[k]; itr++) {
  poly_loop:
    for (unsigned int m = 0; m < N / COEF_PER_BEAT; m++) {
#pragma HLS pipeline
#pragma HLS DEPENDENCE variable = partial_sum_buffer inter false
    PE_loop:
      for (unsigned int cu_id = 0; cu_id < NUM_CU; cu_id++) {
#pragma HLS unroll
        unsigned int bank_id = weight_idx_buffer[cu_id][counter];
        Coef_Bundle in_bundle = act_buffer[bank_id][m];
        Coef_Bundle out_bundle = partial_sum_buffer[cu_id][m];
        for (unsigned i = 0; i < COEF_PER_BEAT; i++) {
          ap_uint<COEF_WIDTH> ps = out_bundle((i + 1) * COEF_WIDTH - 1, i * COEF_WIDTH);
          ap_uint<COEF_WIDTH> in_act = in_bundle((i + 1) * COEF_WIDTH - 1, i * COEF_WIDTH);
          ap_uint<COEF_WIDTH> mult = mod_mult(weight_val_buffer[cu_id][counter], in_act, q_0[rns], q_0_inv[rns]);
          ps = mod_add(ps, mult, q_0[rns]);
          out_bundle((i + 1) * COEF_WIDTH - 1, i * COEF_WIDTH) = ps;
        }
        partial_sum_buffer[cu_id][m] = out_bundle;
      }
    }
    counter++;
  }

  std::cout << "Finished computation." << std::endl;
}

void write_result(
    hls::stream<Coef_Bundle> &pre_act_stream_00,
    hls::stream<Coef_Bundle> &pre_act_stream_01,
    hls::stream<Coef_Bundle> &pre_act_stream_02,
    hls::stream<Coef_Bundle> &pre_act_stream_03,
    hls::stream<Coef_Bundle> &pre_act_stream_04,
    hls::stream<Coef_Bundle> &pre_act_stream_05,
    hls::stream<Coef_Bundle> &pre_act_stream_06,
    hls::stream<Coef_Bundle> &pre_act_stream_07,
    hls::stream<Coef_Bundle> &pre_act_stream_08,
    hls::stream<Coef_Bundle> &pre_act_stream_09,
    hls::stream<Coef_Bundle> &pre_act_stream_10,
    hls::stream<Coef_Bundle> &pre_act_stream_11,
    hls::stream<Coef_Bundle> &pre_act_stream_12,
    hls::stream<Coef_Bundle> &pre_act_stream_13,
    hls::stream<Coef_Bundle> &pre_act_stream_14,
    hls::stream<Coef_Bundle> &pre_act_stream_15,
  Coef_Bundle partial_sum_buffer[NUM_CU][N / COEF_PER_BEAT]) {
  // send out partial sum
  for (unsigned int m = 0; m < N / COEF_PER_BEAT; m++) {
    pre_act_stream_00 << partial_sum_buffer[0][m];
    pre_act_stream_01 << partial_sum_buffer[1][m];
    pre_act_stream_02 << partial_sum_buffer[2][m];
    pre_act_stream_03 << partial_sum_buffer[3][m];
    pre_act_stream_04 << partial_sum_buffer[4][m];
    pre_act_stream_05 << partial_sum_buffer[5][m];
    pre_act_stream_06 << partial_sum_buffer[6][m];
    pre_act_stream_07 << partial_sum_buffer[7][m];
    pre_act_stream_08 << partial_sum_buffer[8][m];
    pre_act_stream_09 << partial_sum_buffer[9][m];
    pre_act_stream_10 << partial_sum_buffer[10][m];
    pre_act_stream_11 << partial_sum_buffer[11][m];
    pre_act_stream_12 << partial_sum_buffer[12][m];
    pre_act_stream_13 << partial_sum_buffer[13][m];
    pre_act_stream_14 << partial_sum_buffer[14][m];
    pre_act_stream_15 << partial_sum_buffer[15][m];
  }
}

void load_act(
    // In act
    Coef_Bundle in_act_0[NUM_CU_PER_BANK * CIN_PER_CU * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle in_act_1[NUM_CU_PER_BANK * CIN_PER_CU * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle in_act_2[NUM_CU_PER_BANK * CIN_PER_CU * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle in_act_3[NUM_CU_PER_BANK * CIN_PER_CU * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle act_buffer[NUM_CU][N / COEF_PER_BEAT], unsigned int k,
    unsigned int rns) {
  // load one polynomial from act iteration k, RNS term rns
  unsigned int base_offset = k * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT +
    rns * N / COEF_PER_BEAT;
  unsigned int factor =
      CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT;
  // std::cout << "load act idx: " << base_offset << std::endl;
  load_act_loop: for (unsigned int m = 0; m < N / COEF_PER_BEAT; m++) {
#pragma HLS pipeline II=1
    unsigned int polynomial_offset = base_offset + m;
    act_buffer[0][m] = in_act_0[0 * factor + polynomial_offset];
    act_buffer[1][m] = in_act_0[1 * factor + polynomial_offset];
    act_buffer[2][m] = in_act_0[2 * factor + polynomial_offset];
    act_buffer[3][m] = in_act_0[3 * factor + polynomial_offset];
    act_buffer[4][m] = in_act_1[0 * factor + polynomial_offset];
    act_buffer[5][m] = in_act_1[1 * factor + polynomial_offset];
    act_buffer[6][m] = in_act_1[2 * factor + polynomial_offset];
    act_buffer[7][m] = in_act_1[3 * factor + polynomial_offset];
    act_buffer[8][m] = in_act_2[0 * factor + polynomial_offset];
    act_buffer[9][m] = in_act_2[1 * factor + polynomial_offset];
    act_buffer[10][m] = in_act_2[2 * factor + polynomial_offset];
    act_buffer[11][m] = in_act_2[3 * factor + polynomial_offset];
    act_buffer[12][m] = in_act_3[0 * factor + polynomial_offset];
    act_buffer[13][m] = in_act_3[1 * factor + polynomial_offset];
    act_buffer[14][m] = in_act_3[2 * factor + polynomial_offset];
    act_buffer[15][m] = in_act_3[3 * factor + polynomial_offset];
  }
  std::cout << "finished writting " << NUM_CU << " polynomials." << std::endl;
}

void compute_linear(
    ap_uint<PARAM_WIDTH> in_loop_count[MAX_ACT_ITRS],
    // In act
    Coef_Bundle in_act_0[NUM_CU_PER_BANK * CIN_PER_CU * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle in_act_1[NUM_CU_PER_BANK * CIN_PER_CU * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle in_act_2[NUM_CU_PER_BANK * CIN_PER_CU * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle in_act_3[NUM_CU_PER_BANK * CIN_PER_CU * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    // pre_act stream
    hls::stream<Coef_Bundle> &pre_act_stream_00,
    hls::stream<Coef_Bundle> &pre_act_stream_01,
    hls::stream<Coef_Bundle> &pre_act_stream_02,
    hls::stream<Coef_Bundle> &pre_act_stream_03,
    hls::stream<Coef_Bundle> &pre_act_stream_04,
    hls::stream<Coef_Bundle> &pre_act_stream_05,
    hls::stream<Coef_Bundle> &pre_act_stream_06,
    hls::stream<Coef_Bundle> &pre_act_stream_07,
    hls::stream<Coef_Bundle> &pre_act_stream_08,
    hls::stream<Coef_Bundle> &pre_act_stream_09,
    hls::stream<Coef_Bundle> &pre_act_stream_10,
    hls::stream<Coef_Bundle> &pre_act_stream_11,
    hls::stream<Coef_Bundle> &pre_act_stream_12,
    hls::stream<Coef_Bundle> &pre_act_stream_13,
    hls::stream<Coef_Bundle> &pre_act_stream_14,
    hls::stream<Coef_Bundle> &pre_act_stream_15,
    // Weight values
    ap_uint<PARAM_WIDTH> NNZ_0[NUM_CU_PER_BANK * MAX_ACT_ITRS],
    ap_uint<PARAM_WIDTH> NNZ_1[NUM_CU_PER_BANK * MAX_ACT_ITRS],
    ap_uint<PARAM_WIDTH> NNZ_2[NUM_CU_PER_BANK * MAX_ACT_ITRS],
    ap_uint<PARAM_WIDTH> NNZ_3[NUM_CU_PER_BANK * MAX_ACT_ITRS],
    // Weight values
    ap_uint<PARAM_WIDTH> weight_values_0[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_1[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_2[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_3[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS],
    // Weight indices
    ap_uint<PARAM_WIDTH> weight_indices_0[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_1[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_2[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_3[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS]) {

  static bool reset = true;
  // double activation buffer to store 1 polynomial each
  Coef_Bundle in_act_buffer_a[NUM_CU][N / COEF_PER_BEAT];
#pragma HLS array_partition variable=in_act_buffer_a cyclic factor=16 dim=1
#pragma HLS bind_storage variable=in_act_buffer_a type=RAM_2P impl=URAM
  Coef_Bundle in_act_buffer_b[NUM_CU][N / COEF_PER_BEAT];
#pragma HLS array_partition variable=in_act_buffer_b cyclic factor=16 dim=1
#pragma HLS bind_storage variable=in_act_buffer_b type=RAM_2P impl=URAM

  // double partial sum buffer to store 1 polynomial each
  Coef_Bundle partial_sum_buffer_a[NUM_CU][N / COEF_PER_BEAT];
#pragma HLS array_partition variable=partial_sum_buffer_a cyclic factor=16 dim=1
#pragma HLS bind_storage variable=partial_sum_buffer_a type=RAM_2P impl=URAM
  Coef_Bundle partial_sum_buffer_b[NUM_CU][N / COEF_PER_BEAT];
#pragma HLS array_partition variable=partial_sum_buffer_b cyclic factor=16 dim=1
#pragma HLS bind_storage variable=partial_sum_buffer_b type=RAM_2P impl=URAM

  // weight buffer to store the entire weights for a layer
  static ap_uint<PARAM_WIDTH> weight_val_buffer[NUM_CU][ON_CHIP_W_MAX_ROWS];
#pragma HLS array_partition variable=weight_val_buffer cyclic factor=16 dim=1
#pragma HLS bind_storage variable=weight_val_buffer type=RAM_2P impl=URAM

	// weight indices buffer to store the entire indices for a layer
  static ap_uint<PARAM_WIDTH> weight_idx_buffer[NUM_CU][ON_CHIP_W_MAX_ROWS];
#pragma HLS array_partition variable=weight_idx_buffer cyclic factor=16 dim=1
#pragma HLS bind_storage variable=weight_idx_buffer type=RAM_2P impl=BRAM

	// NNZ buffer to control the iteration count for each activation read 
  static ap_uint<PARAM_WIDTH> nnz_buffer[MAX_ACT_ITRS];
#pragma HLS bind_storage variable=nnz_buffer type=RAM_2P impl=BRAM
	// NNZ buffer to control the iteration count for each activation read 
  static ap_uint<PARAM_WIDTH> in_loop_count_buffer[MAX_ACT_ITRS];
#pragma HLS array_partition variable=in_loop_count_buffer cyclic factor=2 dim=1
#pragma HLS bind_storage variable=in_loop_count_buffer type=RAM_2P impl=BRAM

  if (reset) {
    // preload all the weights and the indices once for the layer
    for (unsigned int j = 0; j < ON_CHIP_W_MAX_ROWS; j++) {
      weight_val_buffer[0][j] = weight_values_0[0 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[0][j] = weight_indices_0[0 * OFF_CHIP_W_MAX_ROWS + j];

      weight_val_buffer[1][j] = weight_values_0[1 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[1][j] = weight_indices_0[1 * OFF_CHIP_W_MAX_ROWS + j];

      weight_val_buffer[2][j] = weight_values_0[2 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[2][j] = weight_indices_0[2 * OFF_CHIP_W_MAX_ROWS + j];

      weight_val_buffer[3][j] = weight_values_0[3 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[3][j] = weight_indices_0[3 * OFF_CHIP_W_MAX_ROWS + j];

      weight_val_buffer[4][j] = weight_values_1[0 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[4][j] = weight_indices_1[0 * OFF_CHIP_W_MAX_ROWS + j];

      weight_val_buffer[5][j] = weight_values_1[1 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[5][j] = weight_indices_1[1 * OFF_CHIP_W_MAX_ROWS + j];

      weight_val_buffer[6][j] = weight_values_1[2 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[6][j] = weight_indices_1[2 * OFF_CHIP_W_MAX_ROWS + j];

      weight_val_buffer[7][j] = weight_values_1[3 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[7][j] = weight_indices_1[3 * OFF_CHIP_W_MAX_ROWS + j];

      weight_val_buffer[8][j] = weight_values_2[0 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[8][j] = weight_indices_2[0 * OFF_CHIP_W_MAX_ROWS + j];

      weight_val_buffer[9][j] = weight_values_2[1 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[9][j] = weight_indices_2[1 * OFF_CHIP_W_MAX_ROWS + j];

      weight_val_buffer[10][j] = weight_values_2[2 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[10][j] = weight_indices_2[2 * OFF_CHIP_W_MAX_ROWS + j];

      weight_val_buffer[11][j] = weight_values_2[3 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[11][j] = weight_indices_2[3 * OFF_CHIP_W_MAX_ROWS + j];

      weight_val_buffer[12][j] = weight_values_3[0 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[12][j] = weight_indices_3[0 * OFF_CHIP_W_MAX_ROWS + j];

      weight_val_buffer[13][j] = weight_values_3[1 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[13][j] = weight_indices_3[1 * OFF_CHIP_W_MAX_ROWS + j];

      weight_val_buffer[14][j] = weight_values_3[2 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[14][j] = weight_indices_3[2 * OFF_CHIP_W_MAX_ROWS + j];

      weight_val_buffer[15][j] = weight_values_3[3 * OFF_CHIP_W_MAX_ROWS + j];
      weight_idx_buffer[15][j] = weight_indices_3[3 * OFF_CHIP_W_MAX_ROWS + j];
    }
    for (unsigned int j = 0; j < MAX_ACT_ITRS; j++) {
      nnz_buffer[j] = NNZ_0[j];
      in_loop_count_buffer[j] = in_loop_count[j];
    }
    reset = false;
  }

  // First stream cout dimension then stream crt dimension.
  for (unsigned int i = 0; i < R * NUM_CIPHERTEXT_POLY; i++) {
    for (unsigned int j = 0; j < COUT_PER_CU; j++) {
      unsigned int start = (j == 0) ? 0 : static_cast<unsigned int>(in_loop_count[j-1]);
      unsigned int end = in_loop_count[j];
      unsigned int itr_count = end - start;
      std::cout << "iteration count for cout " << j << " batch: " << itr_count << std::endl;

      // for (unsigned int k = 0; k < itr_count; k++) {
      //   load_act(in_act_0, in_act_1, in_act_2, in_act_3, in_act_buffer_a,
      //            k, /*i=*/i);
      //   comp(in_act_buffer_a, partial_sum_buffer_a, weight_val_buffer,
      //        weight_idx_buffer, nnz_buffer, start + k, j, i % R);
      //   write_result(pre_act_stream_00, pre_act_stream_01, pre_act_stream_02,
      //                pre_act_stream_03, pre_act_stream_04, pre_act_stream_05,
      //                pre_act_stream_06, pre_act_stream_07, pre_act_stream_08,
      //                pre_act_stream_09, pre_act_stream_10, pre_act_stream_11,
      //                pre_act_stream_12, pre_act_stream_13, pre_act_stream_14,
      //                pre_act_stream_15, partial_sum_buffer_a);
      // }

      // prologue: load activation buffer
      load_act(in_act_0, in_act_1, in_act_2, in_act_3, in_act_buffer_a,
               /*k=*/0, /*i=*/i);

      // double buffer load / compute / store
      for (unsigned int k = 1; k < itr_count; k++) {
        if (k % 2 == 1) {
          load_act(in_act_0, in_act_1, in_act_2, in_act_3, in_act_buffer_b,
                   /*k=*/k, /*i=*/i);

          if (j % 2 == 0) {
            comp(in_act_buffer_a, partial_sum_buffer_a, weight_val_buffer,
                weight_idx_buffer, nnz_buffer, start + k - 1, j, i % R);
          } else {
            comp(in_act_buffer_a, partial_sum_buffer_b, weight_val_buffer,
                weight_idx_buffer, nnz_buffer, start + k - 1, j, i % R);
          }

        } else {
          load_act(in_act_0, in_act_1, in_act_2, in_act_3, in_act_buffer_a,
                   /*k=*/k, /*i=*/i);

          if (j % 2 == 0) {
            comp(in_act_buffer_b, partial_sum_buffer_a, weight_val_buffer,
                weight_idx_buffer, nnz_buffer, start + k - 1, j, i % R);
          } else {
            comp(in_act_buffer_b, partial_sum_buffer_b, weight_val_buffer,
                weight_idx_buffer, nnz_buffer, start + k - 1, j, i % R);
          }

        }
      }

      // Epilogue: finish the compute of the last compute iteration.
      if (itr_count % 2 == 1) {
        if (j % 2 == 0) {
          comp(in_act_buffer_a, partial_sum_buffer_a, weight_val_buffer,
              weight_idx_buffer, nnz_buffer, start + itr_count - 1, j, i % R);
        } else {
          comp(in_act_buffer_a, partial_sum_buffer_b, weight_val_buffer,
              weight_idx_buffer, nnz_buffer, start + itr_count - 1, j, i % R);
        }
      } else {
        if (j % 2 == 0) {
          comp(in_act_buffer_b, partial_sum_buffer_a, weight_val_buffer,
              weight_idx_buffer, nnz_buffer, start + itr_count - 1, j, i % R);
        } else {
          comp(in_act_buffer_b, partial_sum_buffer_b, weight_val_buffer,
              weight_idx_buffer, nnz_buffer, start + itr_count - 1, j, i % R);
        }
      }

      if (j % 2 == 0) {
        write_result(pre_act_stream_00, pre_act_stream_01, pre_act_stream_02,
                     pre_act_stream_03, pre_act_stream_04, pre_act_stream_05,
                     pre_act_stream_06, pre_act_stream_07, pre_act_stream_08,
                     pre_act_stream_09, pre_act_stream_10, pre_act_stream_11,
                     pre_act_stream_12, pre_act_stream_13, pre_act_stream_14,
                     pre_act_stream_15, partial_sum_buffer_a);
      } else {
        write_result(pre_act_stream_00, pre_act_stream_01, pre_act_stream_02,
                     pre_act_stream_03, pre_act_stream_04, pre_act_stream_05,
                     pre_act_stream_06, pre_act_stream_07, pre_act_stream_08,
                     pre_act_stream_09, pre_act_stream_10, pre_act_stream_11,
                     pre_act_stream_12, pre_act_stream_13, pre_act_stream_14,
                     pre_act_stream_15, partial_sum_buffer_b);
      }
    }
  }

//  // testing
//  unsigned int factor = COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT;
//  testing_loop: for (unsigned int j = 0;
//       j < COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT; j++) {
// #pragma HLS pipeline II=1
//    pre_act_stream_00 << in_act_0[0 * factor + j];
//    pre_act_stream_01 << in_act_0[1 * factor + j];
//    pre_act_stream_02 << in_act_0[2 * factor + j];
//    pre_act_stream_03 << in_act_0[3 * factor + j];
//    pre_act_stream_04 << in_act_1[0 * factor + j];
//    pre_act_stream_05 << in_act_1[1 * factor + j];
//    pre_act_stream_06 << in_act_1[2 * factor + j];
//    pre_act_stream_07 << in_act_1[3 * factor + j];
//    pre_act_stream_08 << in_act_2[0 * factor + j];
//    pre_act_stream_09 << in_act_2[1 * factor + j];
//    pre_act_stream_10 << in_act_2[2 * factor + j];
//    pre_act_stream_11 << in_act_2[3 * factor + j];
//    pre_act_stream_12 << in_act_3[0 * factor + j];
//    pre_act_stream_13 << in_act_3[1 * factor + j];
//    pre_act_stream_14 << in_act_3[2 * factor + j];
//    pre_act_stream_15 << in_act_3[3 * factor + j];
//  }
}

void store_act(
    // pre_act stream
    hls::stream<Coef_Bundle> &pre_act_stream_00,
    hls::stream<Coef_Bundle> &pre_act_stream_01,
    hls::stream<Coef_Bundle> &pre_act_stream_02,
    hls::stream<Coef_Bundle> &pre_act_stream_03,
    hls::stream<Coef_Bundle> &pre_act_stream_04,
    hls::stream<Coef_Bundle> &pre_act_stream_05,
    hls::stream<Coef_Bundle> &pre_act_stream_06,
    hls::stream<Coef_Bundle> &pre_act_stream_07,
    hls::stream<Coef_Bundle> &pre_act_stream_08,
    hls::stream<Coef_Bundle> &pre_act_stream_09,
    hls::stream<Coef_Bundle> &pre_act_stream_10,
    hls::stream<Coef_Bundle> &pre_act_stream_11,
    hls::stream<Coef_Bundle> &pre_act_stream_12,
    hls::stream<Coef_Bundle> &pre_act_stream_13,
    hls::stream<Coef_Bundle> &pre_act_stream_14,
    hls::stream<Coef_Bundle> &pre_act_stream_15,
    // out act
    Coef_Bundle
        out_act_0[NUM_CU_PER_BANK * COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_1[NUM_CU_PER_BANK * COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_2[NUM_CU_PER_BANK * COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_3[NUM_CU_PER_BANK * COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT]) {

  unsigned int factor = COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT;
  out_to_dram_loop: for (unsigned int j = 0;
       j < COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT; j++) {
#pragma HLS pipeline II=1
    out_act_0[0 * factor + j] = pre_act_stream_00.read();
    out_act_0[1 * factor + j] = pre_act_stream_01.read();
    out_act_0[2 * factor + j] = pre_act_stream_02.read();
    out_act_0[3 * factor + j] = pre_act_stream_03.read();
    out_act_1[0 * factor + j] = pre_act_stream_04.read();
    out_act_1[1 * factor + j] = pre_act_stream_05.read();
    out_act_1[2 * factor + j] = pre_act_stream_06.read();
    out_act_1[3 * factor + j] = pre_act_stream_07.read();
    out_act_2[0 * factor + j] = pre_act_stream_08.read();
    out_act_2[1 * factor + j] = pre_act_stream_09.read();
    out_act_2[2 * factor + j] = pre_act_stream_10.read();
    out_act_2[3 * factor + j] = pre_act_stream_11.read();
    out_act_3[0 * factor + j] = pre_act_stream_12.read();
    out_act_3[1 * factor + j] = pre_act_stream_13.read();
    out_act_3[2 * factor + j] = pre_act_stream_14.read();
    out_act_3[3 * factor + j] = pre_act_stream_15.read();
  }
}

// void
// load_act_sparse(Coef_Bundle in_act_00[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     Coef_Bundle in_act_01[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     Coef_Bundle in_act_02[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     Coef_Bundle in_act_03[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     Coef_Bundle in_act_04[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     Coef_Bundle in_act_05[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     Coef_Bundle in_act_06[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     Coef_Bundle in_act_07[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     Coef_Bundle in_act_08[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     Coef_Bundle in_act_09[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     Coef_Bundle in_act_10[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     Coef_Bundle in_act_11[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     Coef_Bundle in_act_12[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     Coef_Bundle in_act_13[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     Coef_Bundle in_act_14[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     Coef_Bundle in_act_15[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
//                           COEF_PER_BEAT],
//     hls::stream<Coef_Bundle> &in_stream_00,
//     hls::stream<Coef_Bundle> &in_stream_01,
//     hls::stream<Coef_Bundle> &in_stream_02,
//     hls::stream<Coef_Bundle> &in_stream_03,
//     hls::stream<Coef_Bundle> &in_stream_04,
//     hls::stream<Coef_Bundle> &in_stream_05,
//     hls::stream<Coef_Bundle> &in_stream_06,
//     hls::stream<Coef_Bundle> &in_stream_07,
//     hls::stream<Coef_Bundle> &in_stream_08,
//     hls::stream<Coef_Bundle> &in_stream_09,
//     hls::stream<Coef_Bundle> &in_stream_10,
//     hls::stream<Coef_Bundle> &in_stream_11,
//     hls::stream<Coef_Bundle> &in_stream_12,
//     hls::stream<Coef_Bundle> &in_stream_13,
//     hls::stream<Coef_Bundle> &in_stream_14,
//     hls::stream<Coef_Bundle> &in_stream_15) {
//   // TODO: Polynomials are organized in the following order:
//   // R * NUM_CIPHERTEXT_POLY * CIN_PER_BANK * K_H * K_W
//   // Each activation is loaded COUT_PER_CU times.
//   for (unsigned int n = 0; n < COUT_PER_CU; n++) {
//     for (unsigned int k = 0; k < R * NUM_CIPHERTEXT_POLY * MAX_ACT_ITRS; k++) {
//       for (unsigned int m = 0; m < N / COEF_PER_BEAT; m++) {
//         unsigned int polynomial_offset = N / COEF_PER_BEAT + m;
//         in_stream_00 << in_act_00[polynomial_offset];
//         in_stream_01 << in_act_01[polynomial_offset];
//         in_stream_02 << in_act_02[polynomial_offset];
//         in_stream_03 << in_act_03[polynomial_offset];
//         in_stream_04 << in_act_04[polynomial_offset];
//         in_stream_05 << in_act_05[polynomial_offset];
//         in_stream_06 << in_act_06[polynomial_offset];
//         in_stream_07 << in_act_07[polynomial_offset];
//         in_stream_08 << in_act_08[polynomial_offset];
//         in_stream_09 << in_act_09[polynomial_offset];
//         in_stream_10 << in_act_10[polynomial_offset];
//         in_stream_11 << in_act_11[polynomial_offset];
//         in_stream_12 << in_act_12[polynomial_offset];
//         in_stream_13 << in_act_13[polynomial_offset];
//         in_stream_14 << in_act_14[polynomial_offset];
//         in_stream_15 << in_act_15[polynomial_offset];
//       }
//     }
//   }
// }

extern "C" {

void he_snn(
    ap_uint<PARAM_WIDTH> in_loop_count[MAX_ACT_ITRS],
    // NNZ
    ap_uint<PARAM_WIDTH> NNZ_0[NUM_CU_PER_BANK * MAX_ACT_ITRS],
    ap_uint<PARAM_WIDTH> NNZ_1[NUM_CU_PER_BANK * MAX_ACT_ITRS],
    ap_uint<PARAM_WIDTH> NNZ_2[NUM_CU_PER_BANK * MAX_ACT_ITRS],
    ap_uint<PARAM_WIDTH> NNZ_3[NUM_CU_PER_BANK * MAX_ACT_ITRS],
    // Weight values
    ap_uint<PARAM_WIDTH> weight_values_0[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_1[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_2[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_3[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS],
    // Weight indices
    ap_uint<PARAM_WIDTH>
        weight_indices_0[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS],
    ap_uint<PARAM_WIDTH>
        weight_indices_1[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS],
    ap_uint<PARAM_WIDTH>
        weight_indices_2[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS],
    ap_uint<PARAM_WIDTH>
        weight_indices_3[NUM_CU_PER_BANK * OFF_CHIP_W_MAX_ROWS],
    // In act
    Coef_Bundle in_act_0[NUM_CU_PER_BANK * CIN_PER_CU * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle in_act_1[NUM_CU_PER_BANK * CIN_PER_CU * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle in_act_2[NUM_CU_PER_BANK * CIN_PER_CU * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle in_act_3[NUM_CU_PER_BANK * CIN_PER_CU * K_H * K_W * R *
                         NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    // out act
    Coef_Bundle out_act_0[NUM_CU_PER_BANK * COUT_PER_CU * R *
                          NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle out_act_1[NUM_CU_PER_BANK * COUT_PER_CU * R *
                          NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle out_act_2[NUM_CU_PER_BANK * COUT_PER_CU * R *
                          NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle out_act_3[NUM_CU_PER_BANK * COUT_PER_CU * R *
                          NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT]) {
#pragma HLS INTERFACE m_axi port = in_loop_count bundle = g_weight_1
#pragma HLS INTERFACE m_axi port = NNZ_0 bundle = g_weight_1
#pragma HLS INTERFACE m_axi port = NNZ_1 bundle = g_weight_1
#pragma HLS INTERFACE m_axi port = NNZ_2 bundle = g_weight_1
#pragma HLS INTERFACE m_axi port = NNZ_3 bundle = g_weight_1

#pragma HLS INTERFACE m_axi port = weight_values_0 bundle = g_weight_1
#pragma HLS INTERFACE m_axi port = weight_values_1 bundle = g_weight_1
#pragma HLS INTERFACE m_axi port = weight_values_2 bundle = g_weight_1
#pragma HLS INTERFACE m_axi port = weight_values_3 bundle = g_weight_1

#pragma HLS INTERFACE m_axi port = weight_indices_0 bundle = g_weight_1
#pragma HLS INTERFACE m_axi port = weight_indices_1 bundle = g_weight_1
#pragma HLS INTERFACE m_axi port = weight_indices_2 bundle = g_weight_1
#pragma HLS INTERFACE m_axi port = weight_indices_3 bundle = g_weight_1

#pragma HLS INTERFACE m_axi port = in_act_0 bundle = g_in_act_0
#pragma HLS INTERFACE m_axi port = in_act_1 bundle = g_in_act_1
#pragma HLS INTERFACE m_axi port = in_act_2 bundle = g_in_act_2
#pragma HLS INTERFACE m_axi port = in_act_3 bundle = g_in_act_3

#pragma HLS INTERFACE m_axi port = out_act_0 bundle = g_out_act_0
#pragma HLS INTERFACE m_axi port = out_act_1 bundle = g_out_act_1
#pragma HLS INTERFACE m_axi port = out_act_2 bundle = g_out_act_2
#pragma HLS INTERFACE m_axi port = out_act_3 bundle = g_out_act_3

  // static hls::stream<Coef_Bundle, 32> in_stream_00;
  // static hls::stream<Coef_Bundle, 32> in_stream_01;
  // static hls::stream<Coef_Bundle, 32> in_stream_02;
  // static hls::stream<Coef_Bundle, 32> in_stream_03;
  // static hls::stream<Coef_Bundle, 32> in_stream_04;
  // static hls::stream<Coef_Bundle, 32> in_stream_05;
  // static hls::stream<Coef_Bundle, 32> in_stream_06;
  // static hls::stream<Coef_Bundle, 32> in_stream_07;
  // static hls::stream<Coef_Bundle, 32> in_stream_08;
  // static hls::stream<Coef_Bundle, 32> in_stream_09;
  // static hls::stream<Coef_Bundle, 32> in_stream_10;
  // static hls::stream<Coef_Bundle, 32> in_stream_11;
  // static hls::stream<Coef_Bundle, 32> in_stream_12;
  // static hls::stream<Coef_Bundle, 32> in_stream_13;
  // static hls::stream<Coef_Bundle, 32> in_stream_14;
  // static hls::stream<Coef_Bundle, 32> in_stream_15;

  static hls::stream<Coef_Bundle, 32> pre_act_stream_00;
  static hls::stream<Coef_Bundle, 32> pre_act_stream_01;
  static hls::stream<Coef_Bundle, 32> pre_act_stream_02;
  static hls::stream<Coef_Bundle, 32> pre_act_stream_03;
  static hls::stream<Coef_Bundle, 32> pre_act_stream_04;
  static hls::stream<Coef_Bundle, 32> pre_act_stream_05;
  static hls::stream<Coef_Bundle, 32> pre_act_stream_06;
  static hls::stream<Coef_Bundle, 32> pre_act_stream_07;
  static hls::stream<Coef_Bundle, 32> pre_act_stream_08;
  static hls::stream<Coef_Bundle, 32> pre_act_stream_09;
  static hls::stream<Coef_Bundle, 32> pre_act_stream_10;
  static hls::stream<Coef_Bundle, 32> pre_act_stream_11;
  static hls::stream<Coef_Bundle, 32> pre_act_stream_12;
  static hls::stream<Coef_Bundle, 32> pre_act_stream_13;
  static hls::stream<Coef_Bundle, 32> pre_act_stream_14;
  static hls::stream<Coef_Bundle, 32> pre_act_stream_15;

#pragma HLS dataflow
  // load_act_sparse(
  //     in_act_00, in_act_01, in_act_02, in_act_03, in_act_04, in_act_05,
  //     in_act_06, in_act_07, in_act_08, in_act_09, in_act_10, in_act_11,
  //     in_act_12, in_act_13, in_act_14, in_act_15, in_stream_00, in_stream_01,
  //     in_stream_02, in_stream_03, in_stream_04, in_stream_05, in_stream_06,
  //     in_stream_07, in_stream_08, in_stream_09, in_stream_10, in_stream_11,
  //     in_stream_12, in_stream_13, in_stream_14, in_stream_15);
  compute_linear(in_loop_count, in_act_0, in_act_1, in_act_2, in_act_3,
                 pre_act_stream_00, pre_act_stream_01, pre_act_stream_02,
                 pre_act_stream_03, pre_act_stream_04, pre_act_stream_05,
                 pre_act_stream_06, pre_act_stream_07, pre_act_stream_08,
                 pre_act_stream_09, pre_act_stream_10, pre_act_stream_11,
                 pre_act_stream_12, pre_act_stream_13, pre_act_stream_14,
                 pre_act_stream_15, NNZ_0, NNZ_1, NNZ_2, NNZ_3, weight_values_0,
                 weight_values_1, weight_values_2, weight_values_3,
                 weight_indices_0, weight_indices_1, weight_indices_2,
                 weight_indices_3);
  store_act(pre_act_stream_00, pre_act_stream_01, pre_act_stream_02,
            pre_act_stream_03, pre_act_stream_04, pre_act_stream_05,
            pre_act_stream_06, pre_act_stream_07, pre_act_stream_08,
            pre_act_stream_09, pre_act_stream_10, pre_act_stream_11,
            pre_act_stream_12, pre_act_stream_13, pre_act_stream_14,
            pre_act_stream_15, out_act_0, out_act_1, out_act_2, out_act_3);
}
}
