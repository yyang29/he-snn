############################## Help Section ##############################
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make host HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build host application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""

TARGET := sw_emu
HOST_ARCH := x86
DEVICE := xilinx_u200_xdma_201830_2

SYSROOT := 

include ./utils.mk

XSA := $(call device2xsa, $(DEVICE))
TEMP_DIR := ./_x.$(TARGET).$(XSA)
BUILD_DIR := ./build_dir.$(TARGET).$(XSA)

PACKAGE_OUT = ./package.$(TARGET)

VPP := v++
CMD_ARGS = $(BUILD_DIR)/he_snn.xclbin

CXXFLAGS += -I$(XILINX_XRT)/include/ -I$(XILINX_VIVADO)/include/ -Wall -O3 -g -std=c++11
LDFLAGS += -L$(XILINX_XRT)/lib/ -lpthread -lOpenCL

############################## Setting up Host Variables ##############################
#Include Required Host Source Files
CXXFLAGS += -I./src/xcl2.h
HOST_SRCS += ./src/xcl2.cpp ./src/host.cpp 
# Host compiler global settings
CXXFLAGS += -fmessage-length=0
LDFLAGS += -lrt -lstdc++ --sysroot=$(SYSROOT)

############################## Setting up Kernel Variables ##############################
# Kernel compiler global settings
VPP_FLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps 
ifneq ($(TARGET), hw)
	VPP_FLAGS += -g
endif

EXECUTABLE = ./he_snn_accel
EMCONFIG_DIR = $(TEMP_DIR)

############################## Declaring Binary Containers ##############################
BINARY_CONTAINERS += $(BUILD_DIR)/he_snn.xclbin
BINARY_CONTAINER_he_snn_OBJS += $(TEMP_DIR)/he_snn.xo

############################## Setting Targets ##############################
CP = cp -rf

.PHONY: all clean cleanall docs emconfig
all: check-devices $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig

.PHONY: host
host: $(EXECUTABLE)

.PHONY: build
build: check-vitis $(BINARY_CONTAINERS)

.PHONY: xclbin
xclbin: build

############################## Setting Rules for Binary Containers (Building Kernels) ##############################
$(TEMP_DIR)/he_snn.xo: src/he_snn.cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k he_snn --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(BUILD_DIR)/he_snn.xclbin: $(BINARY_CONTAINER_he_snn_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_FLAGS) -l $(VPP_LDFLAGS) --temp_dir $(TEMP_DIR)  -o'$(BUILD_DIR)/he_snn.link.xclbin' $(+)
	$(VPP) -p $(BUILD_DIR)/he_snn.link.xclbin -t $(TARGET) --platform $(DEVICE) --package.out_dir $(PACKAGE_OUT) -o $(BUILD_DIR)/he_snn.xclbin

############################## Setting Rules for Host (Building Host Executable) ##############################
$(EXECUTABLE): $(HOST_SRCS) | check-xrt
		$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

############################## Setting Essential Checks and Running Rules ##############################
run: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) $(CMD_ARGS)
else
$(EXECUTABLE) $(CMD_ARGS)
endif

############################## Cleaning Rules ##############################
# Cleaning stuff
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) build_dir* sd_card*
	-$(RMDIR) package.*
	-$(RMDIR) _x* *xclbin.run_summary qemu-memory-_* emulation _vimage pl* start_simulation.sh *.xclbin