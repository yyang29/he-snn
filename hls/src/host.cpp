#include "xcl2.h"
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "ap_int.h"
#include "defs.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File> <weight csv path>"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::string filename(argv[2]);
  std::cout << "weight file: " << filename << std::endl;

  // sparsity map
  std::vector<std::vector<ap_uint<PARAM_WIDTH>,
                          aligned_allocator<ap_uint<PARAM_WIDTH>>>>
      NNZ(NUM_MEM_BANKS, std::vector<ap_uint<PARAM_WIDTH>,
                                     aligned_allocator<ap_uint<PARAM_WIDTH>>>(
                             COUT_PER_BANK, 0));

  // weight values
  std::vector<std::vector<ap_uint<PARAM_WIDTH>,
                          aligned_allocator<ap_uint<PARAM_WIDTH>>>>
      weight_values(NUM_MEM_BANKS,
                    std::vector<ap_uint<PARAM_WIDTH>,
                                aligned_allocator<ap_uint<PARAM_WIDTH>>>(
                        COUT_PER_BANK * MAX_ROWS, 0));

  // weight indices
  std::vector<std::vector<ap_uint<PARAM_WIDTH>,
                          aligned_allocator<ap_uint<PARAM_WIDTH>>>>
      weight_indices(NUM_MEM_BANKS,
                     std::vector<ap_uint<PARAM_WIDTH>,
                                 aligned_allocator<ap_uint<PARAM_WIDTH>>>(
                         COUT_PER_BANK * MAX_ROWS, 0));

  // input activations
  std::vector<std::vector<Coef_Bundle, aligned_allocator<Coef_Bundle>>> in_act(
      NUM_MEM_BANKS, std::vector<Coef_Bundle, aligned_allocator<Coef_Bundle>>(
                         CIN_PER_BANK * K_H * K_W * R * NUM_CIPHERTEXT_POLY *
                         N / COEF_PER_BEAT));

  // output activations
  std::vector<std::vector<Coef_Bundle, aligned_allocator<Coef_Bundle>>> out_act(
      NUM_MEM_BANKS,
      std::vector<Coef_Bundle, aligned_allocator<Coef_Bundle>>(
          COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT));

  // Twiddle factor memory depends on Number of CUs
  // std::vector<
  //     std::vector<ap_uint<COEF_WIDTH>, aligned_allocator<ap_uint<COEF_WIDTH>>>>
  //     tf_ntt(NUM_MEM_BANKS, std::vector<ap_uint<COEF_WIDTH>,
  //                                       aligned_allocator<ap_uint<COEF_WIDTH>>>(
  //                               NUM_CU_PER_BANK * N));
  // std::vector<
  //     std::vector<ap_uint<COEF_WIDTH>, aligned_allocator<ap_uint<COEF_WIDTH>>>>
  //     tf_intt(NUM_MEM_BANKS,
  //             std::vector<ap_uint<COEF_WIDTH>,
  //                         aligned_allocator<ap_uint<COEF_WIDTH>>>(
  //                 NUM_CU_PER_BANK * N));

  // Initialize input data
  for (int i = 0; i < NUM_MEM_BANKS; i++) {
    for (int j = 0; j < CIN_PER_BANK * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                            COEF_PER_BEAT;
         j++) {
      for (int k = 0; k < COEF_PER_BEAT; k++) {
        in_act[i][j].data[k] = std::rand();
      }
    }
    for (int j = 0;
         j < COUT_PER_BANK * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT; j++) {
      for (int k = 0; k < COEF_PER_BEAT; k++) {
        out_act[i][j].data[k] = 0;
      }
    }
    // std::generate(tf_ntt[i].begin(), tf_ntt[i].end(), std::rand);
    // std::generate(tf_intt[i].begin(), tf_intt[i].end(), std::rand);
  }

  // parsing weight file
  std::fstream ifs;
  ifs.open(filename);
  int row_id = 0;
  while (true) {
    std::string line;
    double buf;
    getline(ifs, line);
    std::stringstream ss(line, std::ios_base::out | std::ios_base::in |
                                   std::ios_base::binary);

    if (!ifs)
      break;
    if (line[0] == '#' || line.empty())
      continue;

    std::vector<double> row;
    while (ss >> buf)
      row.push_back(buf);

    for (unsigned int i = 0; i < row.size(); i++) {
      int quantized = static_cast<int>((row[i] + 1) / 2 * 255);
      if (quantized != 127) {
        int c_out_id = i % NUM_MEM_BANKS;
        int c_out_offset = i / NUM_MEM_BANKS;
        int current_nnz = NNZ[c_out_id][c_out_offset];
        weight_values[c_out_id][c_out_offset * MAX_ROWS + current_nnz] =
            quantized;
        weight_indices[c_out_id][c_out_offset * MAX_ROWS + current_nnz] =
            row_id;
        NNZ[c_out_id][c_out_offset]++;
      }
    }
    row_id++;
  }

  /*
  for (int i = 0; i < NUM_MEM_BANKS; i++) {
    for (int j = 0; j < COUT_PER_BANK; j++) {
      std::cout << "NNZs " << NNZ[i][j] << std::endl;
      for (int k = 0; k < MAX_ROWS; k++) {
        std::cout << weight_indices[i][j * MAX_ROWS + k] << " ";
      }
      std::cout << std::endl;
    }
  }
  */

  // device pointers
  std::vector<cl_mem_ext_ptr_t> NNZ_ext(NUM_MEM_BANKS);
  std::vector<cl_mem_ext_ptr_t> weight_values_ext(NUM_MEM_BANKS);
  std::vector<cl_mem_ext_ptr_t> weight_indices_ext(NUM_MEM_BANKS);
  std::vector<cl_mem_ext_ptr_t> in_act_ext(NUM_MEM_BANKS);
  std::vector<cl_mem_ext_ptr_t> out_act_ext(NUM_MEM_BANKS);
  // std::vector<cl_mem_ext_ptr_t> tf_ntt_ext(NUM_MEM_BANKS);
  // std::vector<cl_mem_ext_ptr_t> tf_intt_ext(NUM_MEM_BANKS);
  for (int i = 0; i < NUM_MEM_BANKS; i++) {
    if (i == 0) {
      NNZ_ext[i].flags = XCL_MEM_DDR_BANK0;
      weight_values_ext[i].flags = XCL_MEM_DDR_BANK0;
      weight_indices_ext[i].flags = XCL_MEM_DDR_BANK0;
      in_act_ext[i].flags = XCL_MEM_DDR_BANK0;
      out_act_ext[i].flags = XCL_MEM_DDR_BANK0;
      // tf_ntt_ext[i].flags = XCL_MEM_DDR_BANK0;
      // tf_intt_ext[i].flags = XCL_MEM_DDR_BANK0;
    } else if (i == 1) {
      NNZ_ext[i].flags = XCL_MEM_DDR_BANK1;
      weight_values_ext[i].flags = XCL_MEM_DDR_BANK1;
      weight_indices_ext[i].flags = XCL_MEM_DDR_BANK1;
      in_act_ext[i].flags = XCL_MEM_DDR_BANK1;
      out_act_ext[i].flags = XCL_MEM_DDR_BANK1;
      // tf_ntt_ext[i].flags = XCL_MEM_DDR_BANK1;
      // tf_intt_ext[i].flags = XCL_MEM_DDR_BANK1;
    } else if (i == 2) {
      NNZ_ext[i].flags = XCL_MEM_DDR_BANK2;
      weight_values_ext[i].flags = XCL_MEM_DDR_BANK2;
      weight_indices_ext[i].flags = XCL_MEM_DDR_BANK2;
      in_act_ext[i].flags = XCL_MEM_DDR_BANK2;
      out_act_ext[i].flags = XCL_MEM_DDR_BANK2;
      // tf_ntt_ext[i].flags = XCL_MEM_DDR_BANK2;
      // tf_intt_ext[i].flags = XCL_MEM_DDR_BANK2;
    } else if (i == 3) {
      NNZ_ext[i].flags = XCL_MEM_DDR_BANK3;
      weight_values_ext[i].flags = XCL_MEM_DDR_BANK3;
      weight_indices_ext[i].flags = XCL_MEM_DDR_BANK3;
      in_act_ext[i].flags = XCL_MEM_DDR_BANK3;
      out_act_ext[i].flags = XCL_MEM_DDR_BANK3;
      // tf_ntt_ext[i].flags = XCL_MEM_DDR_BANK3;
      // tf_intt_ext[i].flags = XCL_MEM_DDR_BANK3;
    }
    NNZ_ext[i].obj = NNZ[i].data();
    weight_values_ext[i].obj = weight_values[i].data();
    weight_indices_ext[i].obj = weight_indices[i].data();
    in_act_ext[i].obj = in_act[i].data();
    out_act_ext[i].obj = out_act[i].data();
    // tf_ntt_ext[i].obj = tf_ntt[i].data();
    // tf_intt_ext[i].obj = tf_intt[i].data();
  }

  // OPENCL HOST CODE AREA START
  cl_int err;
  std::string krnl_name = "he_snn";
  cl::Kernel krnl_he_snn;

  // get_xil_devices() is a utility API which will find the xilinx
  // platforms and will return list of devices connected to Xilinx platform
  auto devices = xcl::get_xil_devices();
  auto device = devices[0];

  // Creating Context and Command Queue for selected Device
  OCL_CHECK(err, cl::Context context(device, nullptr, nullptr, nullptr, &err));
  OCL_CHECK(err, cl::CommandQueue q(context, device,
                                    CL_QUEUE_PROFILING_ENABLE |
                                        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                    &err));

  // read_binary_file() is a utility API which will load the binaryFile
  // and will return the pointer to file buffer.
  auto fileBuf = xcl::read_binary_file(static_cast<std::string>(argv[1]));
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  cl::Program program(context, {device}, bins, nullptr, &err);

  // Create kernels specific to compute unit.
  OCL_CHECK(err, krnl_he_snn = cl::Kernel(program, krnl_name.c_str(), &err));

  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  std::vector<cl::Buffer> buffer_NNZ(NUM_MEM_BANKS);
  std::vector<cl::Buffer> buffer_weight_values(NUM_MEM_BANKS);
  std::vector<cl::Buffer> buffer_weight_indices(NUM_MEM_BANKS);
  std::vector<cl::Buffer> buffer_in_act(NUM_MEM_BANKS);
  std::vector<cl::Buffer> buffer_out_act(NUM_MEM_BANKS);
  // std::vector<cl::Buffer> buffer_tf_ntt(NUM_MEM_BANKS);
  // std::vector<cl::Buffer> buffer_tf_intt(NUM_MEM_BANKS);

  for (int i = 0; i < NUM_MEM_BANKS; i++) {
    // read-only buffers
    OCL_CHECK(
        err, buffer_NNZ[i] = cl::Buffer(
                 context,
                 CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                 COUT_PER_BANK * BYTES_INT16, &NNZ_ext[i], &err));
    OCL_CHECK(err, buffer_weight_values[i] =
                       cl::Buffer(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY |
                                      CL_MEM_EXT_PTR_XILINX,
                                  COUT_PER_BANK * MAX_ROWS * BYTES_INT16,
                                  &weight_values_ext[i], &err));
    OCL_CHECK(err, buffer_weight_indices[i] =
                       cl::Buffer(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY |
                                      CL_MEM_EXT_PTR_XILINX,
                                  COUT_PER_BANK * MAX_ROWS * BYTES_INT16,
                                  &weight_indices_ext[i], &err));
    OCL_CHECK(
        err, buffer_in_act[i] = cl::Buffer(
                 context,
                 CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                 CIN_PER_BANK * K_H * K_W * CIPHERTEXT * BYTES_INT64,
                 &in_act_ext[i], &err));
    // OCL_CHECK(
    //     err, buffer_tf_ntt[i] = cl::Buffer(
    //              context,
    //              CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
    //              NUM_CU_PER_BANK * N * BYTES_INT64, &tf_ntt_ext[i], &err));
    // OCL_CHECK(
    //     err, buffer_tf_intt[i] = cl::Buffer(
    //              context,
    //              CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
    //              NUM_CU_PER_BANK * N * BYTES_INT64, &tf_intt_ext[i], &err));
    // write-only buffers
    OCL_CHECK(err, buffer_out_act[i] =
                       cl::Buffer(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY |
                                      CL_MEM_EXT_PTR_XILINX,
                                  COUT_PER_BANK * CIPHERTEXT * BYTES_INT64,
                                  &out_act_ext[i], &err));
  }

  // Using events to ensure dependencies
  std::vector<cl::Event> NNZ_write_event(NUM_MEM_BANKS);
  std::vector<cl::Event> weight_values_write_event(NUM_MEM_BANKS);
  std::vector<cl::Event> weight_indices_write_event(NUM_MEM_BANKS);
  std::vector<cl::Event> in_act_write_event(NUM_MEM_BANKS);
  //std::vector<cl::Event> tf_ntt_write_event(NUM_MEM_BANKS);
  //std::vector<cl::Event> tf_intt_write_event(NUM_MEM_BANKS);
  std::vector<cl::Event> out_act_read_event(NUM_MEM_BANKS);
  std::vector<cl::Event> waiting_events;
  cl::Event task_event;

  // Copy input data to device global memory
  for (int i = 0; i < NUM_MEM_BANKS; i++) {
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_NNZ[i]},
                                                    0 /* 0 means from host*/,
                                                    NULL, &NNZ_write_event[i]));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
                       {buffer_weight_values[i]}, 0 /* 0 means from host*/,
                       NULL, &weight_values_write_event[i]));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
                       {buffer_weight_indices[i]}, 0 /* 0 means from host*/,
                       NULL, &weight_indices_write_event[i]));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
                       {buffer_in_act[i]}, 0 /* 0 means from host*/, NULL,
                       &in_act_write_event[i]));
    // OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
    //                    {buffer_tf_ntt[i]}, 0 /* 0 means from host*/, NULL,
    //                    &tf_ntt_write_event[i]));
    // OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
    //                    {buffer_tf_intt[i]}, 0 /* 0 means from host*/, NULL,
    //                    &tf_intt_write_event[i]));
    waiting_events.push_back(NNZ_write_event[i]);
  }

  // Copy kernel arguments
  int num_args = 0;
  for (int i = 0; i < NUM_MEM_BANKS; i++) {
    OCL_CHECK(err, err = krnl_he_snn.setArg(num_args++, buffer_NNZ[i]));
    OCL_CHECK(err,
              err = krnl_he_snn.setArg(num_args++, buffer_weight_values[i]));
    OCL_CHECK(err,
              err = krnl_he_snn.setArg(num_args++, buffer_weight_indices[i]));
    OCL_CHECK(err, err = krnl_he_snn.setArg(num_args++, buffer_in_act[i]));
    // OCL_CHECK(err, err = krnl_he_snn.setArg(num_args++, buffer_tf_ntt[i]));
    // OCL_CHECK(err, err = krnl_he_snn.setArg(num_args++, buffer_tf_intt[i]));
    OCL_CHECK(err, err = krnl_he_snn.setArg(num_args++, buffer_out_act[i]));
  }

  // Launch the Kernel
  // For HLS kernels global and local size is always (1,1,1). So, it is
  // recommended to always use enqueueTask() for invoking HLS kernel
  OCL_CHECK(err,
            err = q.enqueueTask(krnl_he_snn, &waiting_events, &task_event));
  waiting_events.push_back(task_event);

  for (int i = 0; i < NUM_MEM_BANKS; i++) {
    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
                       {buffer_out_act[i]}, CL_MIGRATE_MEM_OBJECT_HOST,
                       &waiting_events, &out_act_read_event[i]));
  }

  // Wait kernel to finish
  for (int i = 0; i < NUM_MEM_BANKS; i++) {
    out_act_read_event[i].wait();
  }
  // OPENCL HOST CODE AREA END

  // Compare the results of the Device to the simulation
  bool match = true;
  for (int i = 0; i < NUM_MEM_BANKS; i++) {
    for (int j = 0; j < COEF_PER_BEAT; j++) {
      if (out_act[i][0].data[j] != in_act[i][0].data[j]) {
        std::cout << "Result mismatch: ";
        std::cout << out_act[i][0].data[j] << " " << in_act[i][0].data[j]
                  << std::endl;
        match = false;
        break;
      }
    }
  }

  std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
  return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
