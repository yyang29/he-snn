#include "assert.h"
// #include <hls_stream.h>
// #include <hls_vector.h>

#include "ap_int.h"
#include "layer_defs.h"

// #define DATA_SIZE 4096

// // TRIPCOUNT identifier
// const int c_size = DATA_SIZE;

// static void load_input(hls::vector<unsigned int, 16> *in,
//                        hls::stream<hls::vector<unsigned int, 16>> &inStream,
//                        int vSize) {
// mem_rd:
//   for (int i = 0; i < vSize; i++) {
// #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
//     inStream << in[i];
//   }
// }

// static void compute_add(hls::stream<hls::vector<unsigned int, 16>> &in1_stream,
//                         hls::stream<hls::vector<unsigned int, 16>> &in2_stream,
//                         hls::stream<hls::vector<unsigned int, 16>> &out_stream,
//                         int vSize) {
// // The kernel is operating with SIMD vectors of 16 integers. The + operator
// // performs an element-wise add, resulting in 16 parallel additions.
// execute:
//   for (int i = 0; i < vSize; i++) {
// #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
//     out_stream << (in1_stream.read() + in2_stream.read());
//   }
// }

// static void store_result(hls::vector<unsigned int, 16> *out,
//                          hls::stream<hls::vector<unsigned int, 16>> &out_stream,
//                          int vSize) {
// mem_wr:
//   for (int i = 0; i < vSize; i++) {
// #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
//     out[i] = out_stream.read();
//   }
// }

extern "C" {

void he_snn(ap_int<16> NNZ_0[COUT_PER_BANK],
            ap_int<64> out_act_0[COUT_PER_BANK * CIPHERTEXT],
            ap_int<16> NNZ_1[COUT_PER_BANK],
            ap_int<64> out_act_1[COUT_PER_BANK * CIPHERTEXT],
            ap_int<16> NNZ_2[COUT_PER_BANK],
            ap_int<64> out_act_2[COUT_PER_BANK * CIPHERTEXT],
            ap_int<16> NNZ_3[COUT_PER_BANK],
            ap_int<64> out_act_3[COUT_PER_BANK * CIPHERTEXT]
) {
#pragma HLS INTERFACE m_axi port = NNZ_0 bundle = gmem0
#pragma HLS INTERFACE m_axi port = NNZ_1 bundle = gmem1
#pragma HLS INTERFACE m_axi port = NNZ_2 bundle = gmem2
#pragma HLS INTERFACE m_axi port = NNZ_3 bundle = gmem3

#pragma HLS INTERFACE m_axi port = out_act_0 bundle = gmem0
#pragma HLS INTERFACE m_axi port = out_act_1 bundle = gmem1
#pragma HLS INTERFACE m_axi port = out_act_2 bundle = gmem2
#pragma HLS INTERFACE m_axi port = out_act_3 bundle = gmem3
  out_act_0[0] = NNZ_0[0];
  out_act_1[0] = NNZ_1[0];
  out_act_2[0] = NNZ_2[0];
  out_act_3[0] = NNZ_3[0];
}
// /*
//     Vector Addition Kernel

//     Arguments:
//         in1  (input)  --> Input vector 1
//         in2  (input)  --> Input vector 2
//         out  (output) --> Output vector
//         size (input)  --> Number of elements in vector
// */

// void he_snn(hls::vector<unsigned int, 16> *in1,
//             hls::vector<unsigned int, 16> *in2,
//             hls::vector<unsigned int, 16> *out, int size) {
// #pragma HLS INTERFACE m_axi port = in1 bundle = gmem0
// #pragma HLS INTERFACE m_axi port = in2 bundle = gmem1
// #pragma HLS INTERFACE m_axi port = out bundle = gmem0

//   static hls::stream<hls::vector<unsigned int, 16>> in1_stream(
//       "input_stream_1");
//   static hls::stream<hls::vector<unsigned int, 16>> in2_stream(
//       "input_stream_2");
//   static hls::stream<hls::vector<unsigned int, 16>> out_stream("output_stream");

//   // Since 16 values are processed in parallel per loop iteration, the for loop
//   // only needs to iterate 'size / 16' times.
//   assert(size % 16 == 0);
//   int vSize = size / 16;
// #pragma HLS dataflow

//   load_input(in1, in1_stream, vSize);
//   load_input(in2, in2_stream, vSize);
//   compute_add(in1_stream, in2_stream, out_stream, vSize);
//   store_result(out, out_stream, vSize);
// }
}