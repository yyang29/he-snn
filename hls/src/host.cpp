#include "xcl2.h"
#include <algorithm>
#include <vector>

#define DATA_SIZE 4096
#define NUM_CU 4

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }

  // Allocate Memory in Host Memory When creating a buffer with user pointer
  // (CL_MEM_USE_HOST_PTR), under the hood user ptr is used if it is properly
  // aligned. when not aligned, runtime had no choice but to create its own host
  // side buffer. So it is recommended to use this allocator if user wish to
  // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page
  // boundary. It will ensure that user buffer is used when user create
  // Buffer/Mem object with CL_MEM_USE_HOST_PTR
  size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
  std::vector<std::vector<int, aligned_allocator<int>>> source_in1(
      NUM_CU, std::vector<int, aligned_allocator<int>>(DATA_SIZE));
  std::vector<std::vector<int, aligned_allocator<int>>> source_in2(
      NUM_CU, std::vector<int, aligned_allocator<int>>(DATA_SIZE));
  std::vector<std::vector<int, aligned_allocator<int>>> source_hw_results(
      NUM_CU, std::vector<int, aligned_allocator<int>>(DATA_SIZE));
  std::vector<std::vector<int, aligned_allocator<int>>> source_sw_results(
      NUM_CU, std::vector<int, aligned_allocator<int>>(DATA_SIZE));
  std::vector<cl_mem_ext_ptr_t> source_in1_ext(NUM_CU);
  std::vector<cl_mem_ext_ptr_t> source_in2_ext(NUM_CU);
  std::vector<cl_mem_ext_ptr_t> source_hw_results_ext(NUM_CU);

  // Create the test data
  for (int i = 0; i < NUM_CU; i++) {
    std::generate(source_in1[i].begin(), source_in1[i].end(), std::rand);
    std::generate(source_in2[i].begin(), source_in2[i].end(), std::rand);
    for (int j = 0; j < DATA_SIZE; j++) {
      source_sw_results[i][j] = source_in1[i][j] + source_in2[i][j];
      source_hw_results[i][j] = 0;
    }

    // DDR Assignment
    if (i == 0) {
      source_in1_ext[i].flags = XCL_MEM_DDR_BANK0;
      source_in2_ext[i].flags = XCL_MEM_DDR_BANK0;
      source_hw_results_ext[i].flags = XCL_MEM_DDR_BANK0;
    } else if (i == 1) {
      source_in1_ext[i].flags = XCL_MEM_DDR_BANK1;
      source_in2_ext[i].flags = XCL_MEM_DDR_BANK1;
      source_hw_results_ext[i].flags = XCL_MEM_DDR_BANK1;
    } else if (i == 2) {
      source_in1_ext[i].flags = XCL_MEM_DDR_BANK2;
      source_in2_ext[i].flags = XCL_MEM_DDR_BANK2;
      source_hw_results_ext[i].flags = XCL_MEM_DDR_BANK2;
    } else {
      source_in1_ext[i].flags = XCL_MEM_DDR_BANK3;
      source_in2_ext[i].flags = XCL_MEM_DDR_BANK3;
      source_hw_results_ext[i].flags = XCL_MEM_DDR_BANK3;
    }
    source_in1_ext[i].obj = source_in1[i].data();
    source_in2_ext[i].obj = source_in2[i].data();
    source_hw_results_ext[i].obj = source_hw_results[i].data();
  }

  // OPENCL HOST CODE AREA START
  cl_int err;
  std::string krnl_name = "he_snn";
  std::vector<cl::Kernel> krnl_he_snn(NUM_CU);

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
  for (int i = 0; i < NUM_CU; i++) {
    std::string cu_id = std::to_string(i);
    std::string krnl_name_full = krnl_name + ":{" + "he_snn_" + cu_id + "}";
    OCL_CHECK(err, krnl_he_snn[i] = cl::Kernel(program, krnl_name_full.c_str(), &err));
  }

  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  std::vector<cl::Buffer> buffer_in1(NUM_CU), buffer_in2(NUM_CU),
      buffer_output(NUM_CU);

  for (int i = 0; i < NUM_CU; i++) {
    OCL_CHECK(err, buffer_in1[i] =
                       cl::Buffer(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY |
                                      CL_MEM_EXT_PTR_XILINX,
                                  vector_size_bytes, &source_in1_ext[i], &err));
    OCL_CHECK(err, buffer_in2[i] =
                       cl::Buffer(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY |
                                      CL_MEM_EXT_PTR_XILINX,
                                  vector_size_bytes, &source_in2_ext[i], &err));
    OCL_CHECK(err, buffer_output[i] = cl::Buffer(
                       context,
                       CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY |
                           CL_MEM_EXT_PTR_XILINX,
                       vector_size_bytes, &source_hw_results_ext[i], &err));
  }

  // Using events to ensure dependencies
  std::vector<cl::Event> write_event(NUM_CU), read_event(NUM_CU),
      task_event(NUM_CU);
  std::vector<std::vector<cl::Event>> waiting_events(NUM_CU);

  // Copy input data to device global memory
  for (int i = 0; i < NUM_CU; i++) {
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
                       {buffer_in1[i], buffer_in2[i]}, 0 /* 0 means from host*/,
                       NULL, &write_event[i]));
    waiting_events[i].push_back(write_event[i]);
  }

  for (int i = 0; i < NUM_CU; i++) {
    // Launch the Kernel
    // For HLS kernels global and local size is always (1,1,1). So, it is
    // recommended
    // to always use enqueueTask() for invoking HLS kernel
    OCL_CHECK(err, err = krnl_he_snn[i].setArg(0, buffer_in1[i]));
    OCL_CHECK(err, err = krnl_he_snn[i].setArg(1, buffer_in2[i]));
    OCL_CHECK(err, err = krnl_he_snn[i].setArg(2, buffer_output[i]));
    OCL_CHECK(err, err = krnl_he_snn[i].setArg(3, DATA_SIZE));
    OCL_CHECK(err, err = q.enqueueTask(krnl_he_snn[i], &waiting_events[i],
                                       &task_event[i]));
    waiting_events[i].push_back(task_event[i]);
  }

  for (int i = 0; i < NUM_CU; i++) {
    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
                       {buffer_output[i]}, CL_MIGRATE_MEM_OBJECT_HOST,
                       &waiting_events[i], &read_event[i]));
    waiting_events[i].push_back(read_event[i]);
  }

  // Wait kernel to finish
  for (int i = 0; i < NUM_CU; i++) {
    read_event[i].wait();
  }
  // OPENCL HOST CODE AREA END

  // Compare the results of the Device to the simulation
  bool match = true;
  for (int i = 0; i < NUM_CU; i++) {
    for (int j = 0; j < DATA_SIZE; j++) {
      if (source_hw_results[i][j] != source_sw_results[i][j]) {
        std::cout << "Error: CU " << i << " Result mismatch" << std::endl;
        std::cout << "j = " << j << " CPU result = " << source_sw_results[i][j]
                  << " Device result = " << source_hw_results[i][j]
                  << std::endl;
        match = false;
        break;
      }
    }
  }

  std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
  return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}