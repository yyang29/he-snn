#include "xcl2.h"
#include <algorithm>
#include <chrono>
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
      in_loop_count(NUM_CU,
                    std::vector<ap_uint<PARAM_WIDTH>,
                                aligned_allocator<ap_uint<PARAM_WIDTH>>>(
                        MAX_ACT_ITRS, 0));
  // sparsity map
  std::vector<std::vector<ap_uint<PARAM_WIDTH>,
                          aligned_allocator<ap_uint<PARAM_WIDTH>>>>
      NNZ(NUM_CU, std::vector<ap_uint<PARAM_WIDTH>,
                              aligned_allocator<ap_uint<PARAM_WIDTH>>>(
                      MAX_ACT_ITRS, 0));

  // weight values
  std::vector<std::vector<ap_uint<PARAM_WIDTH>,
                          aligned_allocator<ap_uint<PARAM_WIDTH>>>>
      weight_values(NUM_CU,
                    std::vector<ap_uint<PARAM_WIDTH>,
                                aligned_allocator<ap_uint<PARAM_WIDTH>>>(
                        OFF_CHIP_W_MAX_ROWS, 0));

  // weight indices
  std::vector<std::vector<ap_uint<PARAM_WIDTH>,
                          aligned_allocator<ap_uint<PARAM_WIDTH>>>>
      weight_indices(NUM_CU,
                     std::vector<ap_uint<PARAM_WIDTH>,
                                 aligned_allocator<ap_uint<PARAM_WIDTH>>>(
                         OFF_CHIP_W_MAX_ROWS, 0));

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
        in_act[i][j]((k + 1) * COEF_WIDTH - 1, k * COEF_WIDTH) = std::rand();
      }
    }
    for (int j = 0;
         j < COUT_PER_CU * R * NUM_CIPHERTEXT_POLY * N / COEF_PER_BEAT; j++) {
      for (int k = 0; k < COEF_PER_BEAT; k++) {
        out_act[i][j]((k + 1) * COEF_WIDTH - 1, k * COEF_WIDTH) = 0;
      }
    }
  }

  // parsing weight file
  std::fstream ifs;
  ifs.open(filename + "_itr_bp.txt");
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

    unsigned int num = static_cast<unsigned int>(row[0]);
    for (unsigned int i = 0; i < NUM_CU; i++) {
      NNZ[i][row_id] = num;
    }

    row_id++;
  }
  ifs.close();

  ifs.open(filename + "_in_loop_bp.txt");
  row_id = 0;
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

    unsigned int num = static_cast<unsigned int>(row[0]);
    for (unsigned int i = 0; i < NUM_CU; i++) {
      in_loop_count[i][row_id] = num;
    }

    row_id++;
  }
  ifs.close();

  ifs.open(filename + "_wv_bp.txt");
  row_id = 0;
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
      weight_values[i][row_id] = static_cast<unsigned int>(row[i]);
    }
    row_id++;
  }
  ifs.close();

  ifs.open(filename + "_wi_bp.txt");
  row_id = 0;
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
      weight_indices[i][row_id] = static_cast<unsigned int>(row[i]);
    }
    row_id++;
  }
  ifs.close();

  // for (int i = 0; i < OFF_CHIP_W_MAX_ROWS; i++) {
  //   for (int j = 0; j < NUM_CU; j++) {
  //     std::cout << weight_values[j][i] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // for (int i = 0; i < OFF_CHIP_W_MAX_ROWS; i++) {
  //   for (int j = 0; j < NUM_CU; j++) {
  //     std::cout << weight_indices[j][i] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // device pointers
  std::vector<cl_mem_ext_ptr_t> in_loop_count_ext(NUM_CU);
  std::vector<cl_mem_ext_ptr_t> NNZ_ext(NUM_CU);
  std::vector<cl_mem_ext_ptr_t> weight_values_ext(NUM_CU);
  std::vector<cl_mem_ext_ptr_t> weight_indices_ext(NUM_CU);
  std::vector<cl_mem_ext_ptr_t> in_act_ext(NUM_CU);
  std::vector<cl_mem_ext_ptr_t> out_act_ext(NUM_CU);
  for (int i = 0; i < NUM_CU; i++) {
    if (i < 4) {
      in_loop_count_ext[i].flags = XCL_MEM_DDR_BANK0;
      NNZ_ext[i].flags = XCL_MEM_DDR_BANK0;
      weight_values_ext[i].flags = XCL_MEM_DDR_BANK0;
      weight_indices_ext[i].flags = XCL_MEM_DDR_BANK0;
      in_act_ext[i].flags = XCL_MEM_DDR_BANK0;
      out_act_ext[i].flags = XCL_MEM_DDR_BANK0;
    } else if (i < 8) {
      in_loop_count_ext[i].flags = XCL_MEM_DDR_BANK1;
      NNZ_ext[i].flags = XCL_MEM_DDR_BANK1;
      weight_values_ext[i].flags = XCL_MEM_DDR_BANK1;
      weight_indices_ext[i].flags = XCL_MEM_DDR_BANK1;
      in_act_ext[i].flags = XCL_MEM_DDR_BANK1;
      out_act_ext[i].flags = XCL_MEM_DDR_BANK1;
    } else if (i < 12) {
      in_loop_count_ext[i].flags = XCL_MEM_DDR_BANK2;
      NNZ_ext[i].flags = XCL_MEM_DDR_BANK2;
      weight_values_ext[i].flags = XCL_MEM_DDR_BANK2;
      weight_indices_ext[i].flags = XCL_MEM_DDR_BANK2;
      in_act_ext[i].flags = XCL_MEM_DDR_BANK2;
      out_act_ext[i].flags = XCL_MEM_DDR_BANK2;
    } else if (i < 16) {
      in_loop_count_ext[i].flags = XCL_MEM_DDR_BANK3;
      NNZ_ext[i].flags = XCL_MEM_DDR_BANK3;
      weight_values_ext[i].flags = XCL_MEM_DDR_BANK3;
      weight_indices_ext[i].flags = XCL_MEM_DDR_BANK3;
      in_act_ext[i].flags = XCL_MEM_DDR_BANK3;
      out_act_ext[i].flags = XCL_MEM_DDR_BANK3;
    }
    in_loop_count_ext[i].obj = in_loop_count[i].data();
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
  auto start_program = std::chrono::steady_clock::now();
  auto fileBuf = xcl::read_binary_file(static_cast<std::string>(argv[1]));
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  cl::Program program(context, {device}, bins, nullptr, &err);
  auto end_program = std::chrono::steady_clock::now();
  double program_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_program - start_program).count();
  program_time *= 1e-9;
  std::cout << "FPGA program time: " << program_time * 1000 << " msec" << std::endl;;

  // Create kernels specific to compute unit.
  OCL_CHECK(err, krnl_he_snn = cl::Kernel(program, krnl_name.c_str(), &err));

  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  std::vector<cl::Buffer> buffer_in_loop_count(NUM_CU);
  std::vector<cl::Buffer> buffer_NNZ(NUM_CU);
  std::vector<cl::Buffer> buffer_weight_values(NUM_CU);
  std::vector<cl::Buffer> buffer_weight_indices(NUM_CU);
  std::vector<cl::Buffer> buffer_in_act(NUM_CU);
  std::vector<cl::Buffer> buffer_out_act(NUM_CU);

  for (int i = 0; i < NUM_CU; i++) {
    // read-only buffers
    OCL_CHECK(
        err, buffer_in_loop_count[i] = cl::Buffer(
                 context,
                 CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                 MAX_ACT_ITRS * BYTES_INT64, &in_loop_count_ext[i], &err));
    OCL_CHECK(
        err, buffer_NNZ[i] = cl::Buffer(
                 context,
                 CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                 MAX_ACT_ITRS * BYTES_INT64, &NNZ_ext[i], &err));
    OCL_CHECK(err, buffer_weight_values[i] =
                       cl::Buffer(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY |
                                      CL_MEM_EXT_PTR_XILINX,
                                  OFF_CHIP_W_MAX_ROWS * BYTES_INT64,
                                  &weight_values_ext[i], &err));
    OCL_CHECK(err, buffer_weight_indices[i] =
                       cl::Buffer(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY |
                                      CL_MEM_EXT_PTR_XILINX,
                                  OFF_CHIP_W_MAX_ROWS * BYTES_INT64,
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
  std::vector<cl::Event> in_loop_count_write_event(NUM_CU);
  std::vector<cl::Event> NNZ_write_event(NUM_CU);
  std::vector<cl::Event> weight_values_write_event(NUM_CU);
  std::vector<cl::Event> weight_indices_write_event(NUM_CU);
  std::vector<cl::Event> in_act_write_event(NUM_CU);
  std::vector<cl::Event> out_act_read_event(NUM_CU);
  std::vector<cl::Event> waiting_events;
  cl::Event task_event;

  // Copy input data to device global memory
  for (int i = 0; i < NUM_CU; i++) {
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
                       {buffer_in_loop_count[i]}, 0 /* 0 means from host*/,
                       NULL, &in_loop_count_write_event[i]));
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
  OCL_CHECK(err, err = krnl_he_snn.setArg(num_args++, buffer_in_loop_count[0]));
  for (int i = 0; i < NUM_CU; i++) {
    OCL_CHECK(err, err = krnl_he_snn.setArg(num_args++, buffer_NNZ[i]));
  }
  for (int i = 0; i < NUM_CU; i++) {
    OCL_CHECK(err,
              err = krnl_he_snn.setArg(num_args++, buffer_weight_values[i]));
  }
  for (int i = 0; i < NUM_CU; i++) {
    OCL_CHECK(err,
              err = krnl_he_snn.setArg(num_args++, buffer_weight_indices[i]));
  }
  for (int i = 0; i < NUM_CU; i++) {
    OCL_CHECK(err, err = krnl_he_snn.setArg(num_args++, buffer_in_act[i]));
  }
  for (int i = 0; i < NUM_CU; i++) {
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
      if (out_act[i][j] != in_act[i][j]) {
        mismatch_count++;
      }
    }
  }

  std::cout << "mismatch count: " << mismatch_count << std::endl;
  return 0;
}
