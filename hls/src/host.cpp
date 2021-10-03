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
      NNZ(NUM_CU,
          std::vector<ap_uint<PARAM_WIDTH>,
                      aligned_allocator<ap_uint<PARAM_WIDTH>>>(COUT_PER_CU, 0));

  // weight values
  std::vector<std::vector<ap_uint<PARAM_WIDTH>,
                          aligned_allocator<ap_uint<PARAM_WIDTH>>>>
      weight_values(NUM_CU,
                    std::vector<ap_uint<PARAM_WIDTH>,
                                aligned_allocator<ap_uint<PARAM_WIDTH>>>(
                        COUT_PER_CU * MAX_ROWS, 0));

  // weight indices
  std::vector<std::vector<ap_uint<PARAM_WIDTH>,
                          aligned_allocator<ap_uint<PARAM_WIDTH>>>>
      weight_indices(NUM_CU,
                     std::vector<ap_uint<PARAM_WIDTH>,
                                 aligned_allocator<ap_uint<PARAM_WIDTH>>>(
                         COUT_PER_CU * MAX_ROWS, 0));

  // input activations
  std::vector<std::vector<Coef_Bundle, aligned_allocator<Coef_Bundle>>> in_act(
      NUM_CU, std::vector<Coef_Bundle, aligned_allocator<Coef_Bundle>>(
                  CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                  COEF_PER_BEAT));

  // output activations
  std::vector<std::vector<Coef_Bundle, aligned_allocator<Coef_Bundle>>> out_act(
      NUM_CU, std::vector<Coef_Bundle, aligned_allocator<Coef_Bundle>>(
                  COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT));

  // Initialize input data
  for (int i = 0; i < NUM_CU; i++) {
    for (int j = 0; j < CIN_PER_CU * K_H * K_W * R * NUM_CIPHERTEXT_POLY * N /
                            COEF_PER_BEAT;
         j++) {
      for (int k = 0; k < COEF_PER_BEAT; k++) {
        in_act[i][j].data[k] = std::rand();
      }
    }
    for (int j = 0;
         j < COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT; j++) {
      for (int k = 0; k < COEF_PER_BEAT; k++) {
        out_act[i][j].data[k] = 0;
      }
    }
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
        int c_out_id = i % NUM_CU;
        int c_out_offset = i / NUM_CU;
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
  std::vector<cl_mem_ext_ptr_t> NNZ_ext(NUM_CU);
  std::vector<cl_mem_ext_ptr_t> weight_values_ext(NUM_CU);
  std::vector<cl_mem_ext_ptr_t> weight_indices_ext(NUM_CU);
  std::vector<cl_mem_ext_ptr_t> in_act_ext(NUM_CU);
  std::vector<cl_mem_ext_ptr_t> out_act_ext(NUM_CU);
  for (int i = 0; i < NUM_CU; i++) {
    if (i < 4) {
      NNZ_ext[i].flags = XCL_MEM_DDR_BANK0;
      weight_values_ext[i].flags = XCL_MEM_DDR_BANK0;
      weight_indices_ext[i].flags = XCL_MEM_DDR_BANK0;
      in_act_ext[i].flags = XCL_MEM_DDR_BANK0;
      out_act_ext[i].flags = XCL_MEM_DDR_BANK0;
    } else if (i < 8) {
      NNZ_ext[i].flags = XCL_MEM_DDR_BANK1;
      weight_values_ext[i].flags = XCL_MEM_DDR_BANK1;
      weight_indices_ext[i].flags = XCL_MEM_DDR_BANK1;
      in_act_ext[i].flags = XCL_MEM_DDR_BANK1;
      out_act_ext[i].flags = XCL_MEM_DDR_BANK1;
    } else if (i < 12) {
      NNZ_ext[i].flags = XCL_MEM_DDR_BANK2;
      weight_values_ext[i].flags = XCL_MEM_DDR_BANK2;
      weight_indices_ext[i].flags = XCL_MEM_DDR_BANK2;
      in_act_ext[i].flags = XCL_MEM_DDR_BANK2;
      out_act_ext[i].flags = XCL_MEM_DDR_BANK2;
    } else if (i < 16) {
      NNZ_ext[i].flags = XCL_MEM_DDR_BANK3;
      weight_values_ext[i].flags = XCL_MEM_DDR_BANK3;
      weight_indices_ext[i].flags = XCL_MEM_DDR_BANK3;
      in_act_ext[i].flags = XCL_MEM_DDR_BANK3;
      out_act_ext[i].flags = XCL_MEM_DDR_BANK3;
    }
    NNZ_ext[i].obj = NNZ[i].data();
    weight_values_ext[i].obj = weight_values[i].data();
    weight_indices_ext[i].obj = weight_indices[i].data();
    in_act_ext[i].obj = in_act[i].data();
    out_act_ext[i].obj = out_act[i].data();
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
  std::vector<cl::Buffer> buffer_NNZ(NUM_CU);
  std::vector<cl::Buffer> buffer_weight_values(NUM_CU);
  std::vector<cl::Buffer> buffer_weight_indices(NUM_CU);
  std::vector<cl::Buffer> buffer_in_act(NUM_CU);
  std::vector<cl::Buffer> buffer_out_act(NUM_CU);

  for (int i = 0; i < NUM_CU; i++) {
    // read-only buffers
    OCL_CHECK(
        err, buffer_NNZ[i] = cl::Buffer(
                 context,
                 CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                 COUT_PER_CU * BYTES_INT16, &NNZ_ext[i], &err));
    OCL_CHECK(err, buffer_weight_values[i] =
                       cl::Buffer(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY |
                                      CL_MEM_EXT_PTR_XILINX,
                                  COUT_PER_CU * MAX_ROWS * BYTES_INT16,
                                  &weight_values_ext[i], &err));
    OCL_CHECK(err, buffer_weight_indices[i] =
                       cl::Buffer(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY |
                                      CL_MEM_EXT_PTR_XILINX,
                                  COUT_PER_CU * MAX_ROWS * BYTES_INT16,
                                  &weight_indices_ext[i], &err));
    OCL_CHECK(
        err, buffer_in_act[i] = cl::Buffer(
                 context,
                 CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                 CIN_PER_CU * K_H * K_W * CIPHERTEXT * BYTES_INT64,
                 &in_act_ext[i], &err));
    OCL_CHECK(err, buffer_out_act[i] =
                       cl::Buffer(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY |
                                      CL_MEM_EXT_PTR_XILINX,
                                  COUT_PER_CU * CIPHERTEXT * BYTES_INT64,
                                  &out_act_ext[i], &err));
  }

  // Using events to ensure dependencies
  std::vector<cl::Event> NNZ_write_event(NUM_CU);
  std::vector<cl::Event> weight_values_write_event(NUM_CU);
  std::vector<cl::Event> weight_indices_write_event(NUM_CU);
  std::vector<cl::Event> in_act_write_event(NUM_CU);
  std::vector<cl::Event> out_act_read_event(NUM_CU);
  std::vector<cl::Event> waiting_events;
  cl::Event task_event;

  // Copy input data to device global memory
  for (int i = 0; i < NUM_CU; i++) {
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
    waiting_events.push_back(NNZ_write_event[i]);
  }

  // Copy kernel arguments
  int num_args = 0;
  for (int i = 0; i < NUM_CU; i++) {
    OCL_CHECK(err, err = krnl_he_snn.setArg(num_args++, buffer_NNZ[i]));
    OCL_CHECK(err,
              err = krnl_he_snn.setArg(num_args++, buffer_weight_values[i]));
    OCL_CHECK(err,
              err = krnl_he_snn.setArg(num_args++, buffer_weight_indices[i]));
    OCL_CHECK(err, err = krnl_he_snn.setArg(num_args++, buffer_in_act[i]));
    OCL_CHECK(err, err = krnl_he_snn.setArg(num_args++, buffer_out_act[i]));
  }

  // Launch the Kernel
  // For HLS kernels global and local size is always (1,1,1). So, it is
  // recommended to always use enqueueTask() for invoking HLS kernel
  OCL_CHECK(err,
            err = q.enqueueTask(krnl_he_snn, &waiting_events, &task_event));
  waiting_events.push_back(task_event);

  for (int i = 0; i < NUM_CU; i++) {
    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
                       {buffer_out_act[i]}, CL_MIGRATE_MEM_OBJECT_HOST,
                       &waiting_events, &out_act_read_event[i]));
  }

  // Wait kernel to finish
  for (int i = 0; i < NUM_CU; i++) {
    out_act_read_event[i].wait();
  }
  // OPENCL HOST CODE AREA END

  // Compare the results of the Device to the simulation
  unsigned int mismatch_count = 0;
  for (int i = 0; i < NUM_CU; i++) {
    for (int j = 0;
         j < COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT; j++) {
      for (int k = 0; k < COEF_PER_BEAT; k++) {
        if (out_act[i][j].data[k] != in_act[i][j].data[k]) {
          mismatch_count++;
        }
      }
    }
  }

  std::cout << "mismatch count: " << mismatch_count << std::endl;
  return 0;
}
