#include <hls_stream.h>
#include <iostream>

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

void pe_proc(
  Coef_Bundle poly_in_act[N / COEF_PER_BEAT],
  Coef_Bundle poly_out_act[N / COEF_PER_BEAT],
  ap_uint<PARAM_WIDTH> weight_val) {
#pragma hls inline off
  for (unsigned int m = 0; m < N / COEF_PER_BEAT; m++) {
    Coef_Bundle in_bundle = poly_in_act[m];
    Coef_Bundle out_bundle = poly_out_act[m];
    for (unsigned i = 0; i < COEF_PER_BEAT; i++) {
#pragma HLS unroll
      out_bundle.data[i] += mod_mult(weight_val, in_bundle.data[i], q_0, q_0_inv);
    }
    poly_out_act[m] = out_bundle;
  }
}

void comp(
    Coef_Bundle act_buffer[NUM_CU][N / COEF_PER_BEAT],
    Coef_Bundle partial_sum_buffer[NUM_CU][N / COEF_PER_BEAT],
    ap_uint<PARAM_WIDTH> weight_val_buffer[NUM_CU][MAX_COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_idx_buffer[NUM_CU][MAX_COUT_PER_CU * MAX_ROWS],
    unsigned k) {
  static ap_uint<8> current_ptr[NUM_CU];
#pragma HLS ARRAY_PARTITION variable=current_ptr dim=0 complete

  if (k == 0) {
    // reset arbitration
    for (unsigned int i = 0; i < NUM_CU; i++) {
#pragma HLS unroll
      current_ptr[i] = 0;
    }
  }

  unsigned int in_start = k * NUM_CU;
  unsigned int in_end = (k + 1) * NUM_CU;

  bool bank_busy[NUM_CU];
#pragma HLS ARRAY_PARTITION variable=bank_busy dim=0 complete

  bool pe_busy[NUM_CU];
#pragma HLS ARRAY_PARTITION variable=pe_busy dim=0 complete

  int8_t assigned_pe_bank[NUM_CU];
#pragma HLS ARRAY_PARTITION variable=assigned_pe_bank dim=0 complete

  bool done = false;

  while (!done) {
    // arbitration
    for (unsigned int i = 0; i < NUM_CU; i++) {
#pragma HLS pipeline II=1 
      bank_busy[i] = false;
      pe_busy[i] = false;
      assigned_pe_bank[i] = -1;

      ap_uint<8> bank_id = weight_idx_buffer[i][current_ptr[i]] % NUM_CU;
      if (bank_busy[bank_id] == false && current_ptr[i] < in_end) {
        assigned_pe_bank[i] = bank_id;
        bank_busy[bank_id] = true;
        pe_busy[i] = true;
      }
    }

    // computation
    for (unsigned int i = 0; i < NUM_CU; i++) {
#pragma HLS unroll factor=16
#pragma HLS dependence variable=act_buffer inter false
      if (pe_busy[i]) {
        pe_proc(act_buffer[assigned_pe_bank[i]], partial_sum_buffer[i],
           weight_val_buffer[i][current_ptr[i]]);
      }
    }

    // check_done
    done = true;
    for (unsigned int i = 0; i < NUM_CU; i++) {
#pragma HLS pipeline II=1 
      if (pe_busy[i]) current_ptr[i]++;
      if (current_ptr[i] < in_end) {
        done = false;
      }
    }
  }
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
    Coef_Bundle in_act_00[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_01[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_02[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_03[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_04[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_05[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_06[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_07[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_08[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_09[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_10[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_11[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_12[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_13[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_14[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_15[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle act_buffer[NUM_CU][N / COEF_PER_BEAT],
    unsigned int k, unsigned int j) {
  // load one polynomial from act iteration k, RNS term j
  // TODO: Add a flag to skip load activation if all the weights across
  // the cout channels are 0.
  for (unsigned int m = 0; m < N / COEF_PER_BEAT; m++) {
#pragma HLS pipeline II=4
    unsigned int polynomial_offset =
        k * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT +
        j * N / COEF_PER_BEAT + m;
    act_buffer[0][m] = in_act_00[polynomial_offset];
    act_buffer[1][m] = in_act_01[polynomial_offset];
    act_buffer[2][m] = in_act_02[polynomial_offset];
    act_buffer[3][m] = in_act_03[polynomial_offset];
    act_buffer[4][m] = in_act_04[polynomial_offset];
    act_buffer[5][m] = in_act_05[polynomial_offset];
    act_buffer[6][m] = in_act_06[polynomial_offset];
    act_buffer[7][m] = in_act_07[polynomial_offset];
    act_buffer[8][m] = in_act_08[polynomial_offset];
    act_buffer[9][m] = in_act_09[polynomial_offset];
    act_buffer[10][m] = in_act_10[polynomial_offset];
    act_buffer[11][m] = in_act_11[polynomial_offset];
    act_buffer[12][m] = in_act_12[polynomial_offset];
    act_buffer[13][m] = in_act_13[polynomial_offset];
    act_buffer[14][m] = in_act_14[polynomial_offset];
    act_buffer[15][m] = in_act_15[polynomial_offset];
  }
  std::cout << "finished writting " << NUM_CU << " polynomials." << std::endl;
}

void compute_linear(
    // In act
    Coef_Bundle in_act_00[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_01[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_02[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_03[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_04[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_05[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_06[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_07[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_08[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_09[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_10[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_11[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_12[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_13[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_14[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_15[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    // hls::stream<Coef_Bundle> &in_stream_00,
    // hls::stream<Coef_Bundle> &in_stream_01,
    // hls::stream<Coef_Bundle> &in_stream_02,
    // hls::stream<Coef_Bundle> &in_stream_03,
    // hls::stream<Coef_Bundle> &in_stream_04,
    // hls::stream<Coef_Bundle> &in_stream_05,
    // hls::stream<Coef_Bundle> &in_stream_06,
    // hls::stream<Coef_Bundle> &in_stream_07,
    // hls::stream<Coef_Bundle> &in_stream_08,
    // hls::stream<Coef_Bundle> &in_stream_09,
    // hls::stream<Coef_Bundle> &in_stream_10,
    // hls::stream<Coef_Bundle> &in_stream_11,
    // hls::stream<Coef_Bundle> &in_stream_12,
    // hls::stream<Coef_Bundle> &in_stream_13,
    // hls::stream<Coef_Bundle> &in_stream_14,
    // hls::stream<Coef_Bundle> &in_stream_15,
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
    ap_uint<PARAM_WIDTH> weight_values_00[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_01[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_02[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_03[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_04[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_05[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_06[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_07[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_08[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_09[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_10[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_11[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_12[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_13[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_14[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_15[COUT_PER_CU * MAX_ROWS],
    // Weight indices
    ap_uint<PARAM_WIDTH> weight_indices_00[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_01[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_02[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_03[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_04[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_05[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_06[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_07[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_08[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_09[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_10[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_11[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_12[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_13[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_14[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_15[COUT_PER_CU * MAX_ROWS]) {

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
  static ap_uint<PARAM_WIDTH> weight_val_buffer[NUM_CU][MAX_COUT_PER_CU * MAX_ROWS];
#pragma HLS array_partition variable=weight_val_buffer cyclic factor=16 dim=1
#pragma HLS bind_storage variable=weight_val_buffer type=RAM_2P impl=BRAM

	// weight indices buffer to store the entire indices for a layer
  static ap_uint<PARAM_WIDTH> weight_idx_buffer[NUM_CU][MAX_COUT_PER_CU * MAX_ROWS];
#pragma HLS array_partition variable=weight_idx_buffer cyclic factor=16 dim=1
#pragma HLS bind_storage variable=weight_idx_buffer type=RAM_2P impl=BRAM

  if (reset) {
    // preload all the weights and the indices once for the layer
    for (unsigned int j = 0; j < COUT_PER_CU; j++) {
      for (unsigned int k = 0; k < MAX_ROWS; k++) {
        unsigned int weight_offset = j * MAX_ROWS + k;
        weight_val_buffer[0][weight_offset] = weight_values_00[weight_offset];
        weight_idx_buffer[0][weight_offset] = weight_indices_00[weight_offset];

        weight_val_buffer[1][weight_offset] = weight_values_01[weight_offset];
        weight_idx_buffer[1][weight_offset] = weight_indices_01[weight_offset];

        weight_val_buffer[2][weight_offset] = weight_values_02[weight_offset];
        weight_idx_buffer[2][weight_offset] = weight_indices_02[weight_offset];

        weight_val_buffer[3][weight_offset] = weight_values_03[weight_offset];
        weight_idx_buffer[3][weight_offset] = weight_indices_03[weight_offset];

        weight_val_buffer[4][weight_offset] = weight_values_04[weight_offset];
        weight_idx_buffer[4][weight_offset] = weight_indices_04[weight_offset];

        weight_val_buffer[5][weight_offset] = weight_values_05[weight_offset];
        weight_idx_buffer[5][weight_offset] = weight_indices_05[weight_offset];

        weight_val_buffer[6][weight_offset] = weight_values_06[weight_offset];
        weight_idx_buffer[6][weight_offset] = weight_indices_06[weight_offset];

        weight_val_buffer[7][weight_offset] = weight_values_07[weight_offset];
        weight_idx_buffer[7][weight_offset] = weight_indices_07[weight_offset];

        weight_val_buffer[8][weight_offset] = weight_values_08[weight_offset];
        weight_idx_buffer[8][weight_offset] = weight_indices_08[weight_offset];

        weight_val_buffer[9][weight_offset] = weight_values_09[weight_offset];
        weight_idx_buffer[9][weight_offset] = weight_indices_09[weight_offset];

        weight_val_buffer[10][weight_offset] = weight_values_10[weight_offset];
        weight_idx_buffer[10][weight_offset] = weight_indices_10[weight_offset];

        weight_val_buffer[11][weight_offset] = weight_values_11[weight_offset];
        weight_idx_buffer[11][weight_offset] = weight_indices_11[weight_offset];

        weight_val_buffer[12][weight_offset] = weight_values_12[weight_offset];
        weight_idx_buffer[12][weight_offset] = weight_indices_12[weight_offset];

        weight_val_buffer[13][weight_offset] = weight_values_13[weight_offset];
        weight_idx_buffer[13][weight_offset] = weight_indices_13[weight_offset];

        weight_val_buffer[14][weight_offset] = weight_values_14[weight_offset];
        weight_idx_buffer[14][weight_offset] = weight_indices_14[weight_offset];

        weight_val_buffer[15][weight_offset] = weight_values_15[weight_offset];
        weight_idx_buffer[15][weight_offset] = weight_indices_15[weight_offset];
      }
    }
    reset = false;
  }

  // First stream cout dimension then stream crt dimension.
  for (unsigned int i = 0; i < R * NUM_CIPHERTEXT_POLY; i++) {
    for (unsigned int j = 0; j < COUT_PER_CU; j++) {
      // prologue: load activation buffer
      load_act(in_act_00, in_act_01, in_act_02, in_act_03, in_act_04, in_act_05,
               in_act_06, in_act_07, in_act_08, in_act_09, in_act_10, in_act_11,
               in_act_12, in_act_13, in_act_14, in_act_15, in_act_buffer_a,
               /*k=*/0, /*i=*/i);

      // // double buffer load / compute / store
      // for (unsigned int k = 1; k < MAX_ACT_ITRS; k++) {
      //   if (k % 2 == 1) {
      //     load_act(in_act_00, in_act_01, in_act_02, in_act_03, in_act_04,
      //              in_act_05, in_act_06, in_act_07, in_act_08, in_act_09,
      //              in_act_10, in_act_11, in_act_12, in_act_13, in_act_14,
      //              in_act_15, in_act_buffer_b,
      //              /*k=*/k, /*i=*/i);

      //     if (j % 2 == 0) {
      //       comp(in_act_buffer_a, partial_sum_buffer_a, weight_val_buffer,
      //           weight_idx_buffer, k - 1);
      //     } else {
      //       comp(in_act_buffer_a, partial_sum_buffer_b, weight_val_buffer,
      //           weight_idx_buffer, k - 1);
      //     }

      //   } else {
      //     load_act(in_act_00, in_act_01, in_act_02, in_act_03, in_act_04,
      //              in_act_05, in_act_06, in_act_07, in_act_08, in_act_09,
      //              in_act_10, in_act_11, in_act_12, in_act_13, in_act_14,
      //              in_act_15, in_act_buffer_a,
      //              /*k=*/k, /*i=*/i);

      //     if (j % 2 == 0) {
      //       comp(in_act_buffer_b, partial_sum_buffer_a, weight_val_buffer,
      //           weight_idx_buffer, k - 1);
      //     } else {
      //       comp(in_act_buffer_b, partial_sum_buffer_b, weight_val_buffer,
      //           weight_idx_buffer, k - 1);
      //     }

      //   }
      // }

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

//   // testing
//   testing_loop: for (unsigned int j = 0;
//        j < N / COEF_PER_BEAT; j++) {
// #pragma HLS pipeline II=2
//     pre_act_stream_00 << in_act_buffer_a[0][j];
//     pre_act_stream_01 << in_act_buffer_a[1][j];
//     pre_act_stream_02 << in_act_buffer_a[2][j];
//     pre_act_stream_03 << in_act_buffer_a[3][j];
//     pre_act_stream_04 << in_act_buffer_a[4][j];
//     pre_act_stream_05 << in_act_buffer_a[5][j];
//     pre_act_stream_06 << in_act_buffer_a[6][j];
//     pre_act_stream_07 << in_act_buffer_a[7][j];
//     pre_act_stream_08 << in_act_buffer_a[8][j];
//     pre_act_stream_09 << in_act_buffer_a[9][j];
//     pre_act_stream_10 << in_act_buffer_a[10][j];
//     pre_act_stream_11 << in_act_buffer_a[11][j];
//     pre_act_stream_12 << in_act_buffer_a[12][j];
//     pre_act_stream_13 << in_act_buffer_a[13][j];
//     pre_act_stream_14 << in_act_buffer_a[14][j];
//     pre_act_stream_15 << in_act_buffer_a[15][j];
//   }
}

static void store_act(
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
        out_act_00[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_01[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_02[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_03[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_04[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_05[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_06[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_07[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_08[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_09[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_10[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_11[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_12[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_13[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_14[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_15[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT]) {
  for (unsigned int j = 0;
       j < COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT; j++) {
    out_act_00[j] = pre_act_stream_00.read();
    out_act_01[j] = pre_act_stream_01.read();
    out_act_02[j] = pre_act_stream_02.read();
    out_act_03[j] = pre_act_stream_03.read();
    out_act_04[j] = pre_act_stream_04.read();
    out_act_05[j] = pre_act_stream_05.read();
    out_act_06[j] = pre_act_stream_06.read();
    out_act_07[j] = pre_act_stream_07.read();
    out_act_08[j] = pre_act_stream_08.read();
    out_act_09[j] = pre_act_stream_09.read();
    out_act_10[j] = pre_act_stream_10.read();
    out_act_11[j] = pre_act_stream_11.read();
    out_act_12[j] = pre_act_stream_12.read();
    out_act_13[j] = pre_act_stream_13.read();
    out_act_14[j] = pre_act_stream_14.read();
    out_act_15[j] = pre_act_stream_15.read();
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
    // NNZ
    ap_uint<PARAM_WIDTH> NNZ_00[COUT_PER_CU],
    ap_uint<PARAM_WIDTH> NNZ_01[COUT_PER_CU],
    ap_uint<PARAM_WIDTH> NNZ_02[COUT_PER_CU],
    ap_uint<PARAM_WIDTH> NNZ_03[COUT_PER_CU],
    ap_uint<PARAM_WIDTH> NNZ_04[COUT_PER_CU],
    ap_uint<PARAM_WIDTH> NNZ_05[COUT_PER_CU],
    ap_uint<PARAM_WIDTH> NNZ_06[COUT_PER_CU],
    ap_uint<PARAM_WIDTH> NNZ_07[COUT_PER_CU],
    ap_uint<PARAM_WIDTH> NNZ_08[COUT_PER_CU],
    ap_uint<PARAM_WIDTH> NNZ_09[COUT_PER_CU],
    ap_uint<PARAM_WIDTH> NNZ_10[COUT_PER_CU],
    ap_uint<PARAM_WIDTH> NNZ_11[COUT_PER_CU],
    ap_uint<PARAM_WIDTH> NNZ_12[COUT_PER_CU],
    ap_uint<PARAM_WIDTH> NNZ_13[COUT_PER_CU],
    ap_uint<PARAM_WIDTH> NNZ_14[COUT_PER_CU],
    ap_uint<PARAM_WIDTH> NNZ_15[COUT_PER_CU],
    // Weight values
    ap_uint<PARAM_WIDTH> weight_values_00[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_01[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_02[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_03[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_04[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_05[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_06[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_07[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_08[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_09[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_10[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_11[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_12[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_13[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_14[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_values_15[COUT_PER_CU * MAX_ROWS],
    // Weight indices
    ap_uint<PARAM_WIDTH> weight_indices_00[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_01[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_02[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_03[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_04[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_05[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_06[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_07[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_08[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_09[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_10[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_11[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_12[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_13[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_14[COUT_PER_CU * MAX_ROWS],
    ap_uint<PARAM_WIDTH> weight_indices_15[COUT_PER_CU * MAX_ROWS],
    // In act
    Coef_Bundle in_act_00[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_01[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_02[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_03[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_04[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_05[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_06[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_07[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_08[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_09[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_10[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_11[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_12[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_13[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_14[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    Coef_Bundle in_act_15[CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                          COEF_PER_BEAT],
    // out act
    Coef_Bundle
        out_act_00[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_01[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_02[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_03[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_04[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_05[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_06[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_07[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_08[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_09[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_10[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_11[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_12[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_13[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_14[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT],
    Coef_Bundle
        out_act_15[COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT]) {
#pragma HLS INTERFACE m_axi port = NNZ_00 bundle = g_weight_00
#pragma HLS INTERFACE m_axi port = NNZ_01 bundle = g_weight_00
#pragma HLS INTERFACE m_axi port = NNZ_02 bundle = g_weight_00
#pragma HLS INTERFACE m_axi port = NNZ_03 bundle = g_weight_00
#pragma HLS INTERFACE m_axi port = NNZ_04 bundle = g_weight_01
#pragma HLS INTERFACE m_axi port = NNZ_05 bundle = g_weight_01
#pragma HLS INTERFACE m_axi port = NNZ_06 bundle = g_weight_01
#pragma HLS INTERFACE m_axi port = NNZ_07 bundle = g_weight_01
#pragma HLS INTERFACE m_axi port = NNZ_08 bundle = g_weight_02
#pragma HLS INTERFACE m_axi port = NNZ_09 bundle = g_weight_02
#pragma HLS INTERFACE m_axi port = NNZ_10 bundle = g_weight_02
#pragma HLS INTERFACE m_axi port = NNZ_11 bundle = g_weight_02
#pragma HLS INTERFACE m_axi port = NNZ_12 bundle = g_weight_03
#pragma HLS INTERFACE m_axi port = NNZ_13 bundle = g_weight_03
#pragma HLS INTERFACE m_axi port = NNZ_14 bundle = g_weight_03
#pragma HLS INTERFACE m_axi port = NNZ_15 bundle = g_weight_03

#pragma HLS INTERFACE m_axi port = weight_values_00 bundle = g_weight_00
#pragma HLS INTERFACE m_axi port = weight_values_01 bundle = g_weight_00
#pragma HLS INTERFACE m_axi port = weight_values_02 bundle = g_weight_00
#pragma HLS INTERFACE m_axi port = weight_values_03 bundle = g_weight_00
#pragma HLS INTERFACE m_axi port = weight_values_04 bundle = g_weight_01
#pragma HLS INTERFACE m_axi port = weight_values_05 bundle = g_weight_01
#pragma HLS INTERFACE m_axi port = weight_values_06 bundle = g_weight_01
#pragma HLS INTERFACE m_axi port = weight_values_07 bundle = g_weight_01
#pragma HLS INTERFACE m_axi port = weight_values_08 bundle = g_weight_02
#pragma HLS INTERFACE m_axi port = weight_values_09 bundle = g_weight_02
#pragma HLS INTERFACE m_axi port = weight_values_10 bundle = g_weight_02
#pragma HLS INTERFACE m_axi port = weight_values_11 bundle = g_weight_02
#pragma HLS INTERFACE m_axi port = weight_values_12 bundle = g_weight_03
#pragma HLS INTERFACE m_axi port = weight_values_13 bundle = g_weight_03
#pragma HLS INTERFACE m_axi port = weight_values_14 bundle = g_weight_03
#pragma HLS INTERFACE m_axi port = weight_values_15 bundle = g_weight_03

#pragma HLS INTERFACE m_axi port = weight_indices_00 bundle = g_weight_00
#pragma HLS INTERFACE m_axi port = weight_indices_01 bundle = g_weight_00
#pragma HLS INTERFACE m_axi port = weight_indices_02 bundle = g_weight_00
#pragma HLS INTERFACE m_axi port = weight_indices_03 bundle = g_weight_00
#pragma HLS INTERFACE m_axi port = weight_indices_04 bundle = g_weight_01
#pragma HLS INTERFACE m_axi port = weight_indices_05 bundle = g_weight_01
#pragma HLS INTERFACE m_axi port = weight_indices_06 bundle = g_weight_01
#pragma HLS INTERFACE m_axi port = weight_indices_07 bundle = g_weight_01
#pragma HLS INTERFACE m_axi port = weight_indices_08 bundle = g_weight_02
#pragma HLS INTERFACE m_axi port = weight_indices_09 bundle = g_weight_02
#pragma HLS INTERFACE m_axi port = weight_indices_10 bundle = g_weight_02
#pragma HLS INTERFACE m_axi port = weight_indices_11 bundle = g_weight_02
#pragma HLS INTERFACE m_axi port = weight_indices_12 bundle = g_weight_03
#pragma HLS INTERFACE m_axi port = weight_indices_13 bundle = g_weight_03
#pragma HLS INTERFACE m_axi port = weight_indices_14 bundle = g_weight_03
#pragma HLS INTERFACE m_axi port = weight_indices_15 bundle = g_weight_03

#pragma HLS INTERFACE m_axi port = in_act_00 bundle = g_in_act_00
#pragma HLS INTERFACE m_axi port = in_act_01 bundle = g_in_act_01
#pragma HLS INTERFACE m_axi port = in_act_02 bundle = g_in_act_02
#pragma HLS INTERFACE m_axi port = in_act_03 bundle = g_in_act_03
#pragma HLS INTERFACE m_axi port = in_act_04 bundle = g_in_act_04
#pragma HLS INTERFACE m_axi port = in_act_05 bundle = g_in_act_05
#pragma HLS INTERFACE m_axi port = in_act_06 bundle = g_in_act_06
#pragma HLS INTERFACE m_axi port = in_act_07 bundle = g_in_act_07
#pragma HLS INTERFACE m_axi port = in_act_08 bundle = g_in_act_08
#pragma HLS INTERFACE m_axi port = in_act_09 bundle = g_in_act_09
#pragma HLS INTERFACE m_axi port = in_act_10 bundle = g_in_act_10
#pragma HLS INTERFACE m_axi port = in_act_11 bundle = g_in_act_11
#pragma HLS INTERFACE m_axi port = in_act_12 bundle = g_in_act_12
#pragma HLS INTERFACE m_axi port = in_act_13 bundle = g_in_act_13
#pragma HLS INTERFACE m_axi port = in_act_14 bundle = g_in_act_14
#pragma HLS INTERFACE m_axi port = in_act_15 bundle = g_in_act_15

#pragma HLS INTERFACE m_axi port = out_act_00 bundle = g_out_act_00
#pragma HLS INTERFACE m_axi port = out_act_01 bundle = g_out_act_01
#pragma HLS INTERFACE m_axi port = out_act_02 bundle = g_out_act_02
#pragma HLS INTERFACE m_axi port = out_act_03 bundle = g_out_act_03
#pragma HLS INTERFACE m_axi port = out_act_04 bundle = g_out_act_04
#pragma HLS INTERFACE m_axi port = out_act_05 bundle = g_out_act_05
#pragma HLS INTERFACE m_axi port = out_act_06 bundle = g_out_act_06
#pragma HLS INTERFACE m_axi port = out_act_07 bundle = g_out_act_07
#pragma HLS INTERFACE m_axi port = out_act_08 bundle = g_out_act_08
#pragma HLS INTERFACE m_axi port = out_act_09 bundle = g_out_act_09
#pragma HLS INTERFACE m_axi port = out_act_10 bundle = g_out_act_10
#pragma HLS INTERFACE m_axi port = out_act_11 bundle = g_out_act_11
#pragma HLS INTERFACE m_axi port = out_act_12 bundle = g_out_act_12
#pragma HLS INTERFACE m_axi port = out_act_13 bundle = g_out_act_13
#pragma HLS INTERFACE m_axi port = out_act_14 bundle = g_out_act_14
#pragma HLS INTERFACE m_axi port = out_act_15 bundle = g_out_act_15

  static hls::stream<Coef_Bundle, 32> in_stream_00;
  static hls::stream<Coef_Bundle, 32> in_stream_01;
  static hls::stream<Coef_Bundle, 32> in_stream_02;
  static hls::stream<Coef_Bundle, 32> in_stream_03;
  static hls::stream<Coef_Bundle, 32> in_stream_04;
  static hls::stream<Coef_Bundle, 32> in_stream_05;
  static hls::stream<Coef_Bundle, 32> in_stream_06;
  static hls::stream<Coef_Bundle, 32> in_stream_07;
  static hls::stream<Coef_Bundle, 32> in_stream_08;
  static hls::stream<Coef_Bundle, 32> in_stream_09;
  static hls::stream<Coef_Bundle, 32> in_stream_10;
  static hls::stream<Coef_Bundle, 32> in_stream_11;
  static hls::stream<Coef_Bundle, 32> in_stream_12;
  static hls::stream<Coef_Bundle, 32> in_stream_13;
  static hls::stream<Coef_Bundle, 32> in_stream_14;
  static hls::stream<Coef_Bundle, 32> in_stream_15;

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
  compute_linear(
      in_act_00, in_act_01, in_act_02, in_act_03, in_act_04, in_act_05,
      in_act_06, in_act_07, in_act_08, in_act_09, in_act_10, in_act_11,
      in_act_12, in_act_13, in_act_14, in_act_15, pre_act_stream_00,
      pre_act_stream_01, pre_act_stream_02, pre_act_stream_03,
      pre_act_stream_04, pre_act_stream_05, pre_act_stream_06,
      pre_act_stream_07, pre_act_stream_08, pre_act_stream_09,
      pre_act_stream_10, pre_act_stream_11, pre_act_stream_12,
      pre_act_stream_13, pre_act_stream_14, pre_act_stream_15, weight_values_00,
      weight_values_01, weight_values_02, weight_values_03, weight_values_04,
      weight_values_05, weight_values_06, weight_values_07, weight_values_08,
      weight_values_09, weight_values_10, weight_values_11, weight_values_12,
      weight_values_13, weight_values_14, weight_values_15, weight_indices_00,
      weight_indices_01, weight_indices_02, weight_indices_03,
      weight_indices_04, weight_indices_05, weight_indices_06,
      weight_indices_07, weight_indices_08, weight_indices_09,
      weight_indices_10, weight_indices_11, weight_indices_12,
      weight_indices_13, weight_indices_14, weight_indices_15);
  store_act(pre_act_stream_00, pre_act_stream_01, pre_act_stream_02,
            pre_act_stream_03, pre_act_stream_04, pre_act_stream_05,
            pre_act_stream_06, pre_act_stream_07, pre_act_stream_08,
            pre_act_stream_09, pre_act_stream_10, pre_act_stream_11,
            pre_act_stream_12, pre_act_stream_13, pre_act_stream_14,
            pre_act_stream_15, out_act_00, out_act_01, out_act_02, out_act_03,
            out_act_04, out_act_05, out_act_06, out_act_07, out_act_08,
            out_act_09, out_act_10, out_act_11, out_act_12, out_act_13,
            out_act_14, out_act_15);
}
}