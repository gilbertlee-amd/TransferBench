#
# Copyright (c) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
#

# Configuration options
ROCM_PATH ?= /opt/rocm
CUDA_PATH ?= /usr/local/cuda

HIPCC=$(ROCM_PATH)/bin/hipcc
NVCC=$(CUDA_PATH)/bin/nvcc

# Compile TransferBenchCuda if nvcc detected
ifeq ("$(shell test -e $(NVCC) && echo found)", "found")
  EXE=TransferBenchCuda
else
  EXE=TransferBench
endif

CXXFLAGS = -I$(ROCM_PATH)/include -lnuma -L$(ROCM_PATH)/lib -lhsa-runtime64
NVFLAGS  = -x cu -lnuma -arch=native
COMMON_FLAGS = -O3 -I./src/header -I./src/client -I./src/client/Presets
LDFLAGS += -lpthread

# Compile RDMA executor if IBVerbs is found in the Dynamic Linker cache
NIC_ENABLED = 0
ifneq ($(DISABLE_NIC_EXEC),1)
  ifneq ("$(shell ldconfig -p | grep -c ibverbs)", "0")
    LDFLAGS += -libverbs -DNIC_EXEC_ENABLED
    NVFLAGS += -libverbs -DNIC_EXEC_ENABLED
    NIC_ENABLED = 1
  endif
endif

all: $(EXE)

TransferBench: ./src/client/Client.cpp $(shell find -regex ".*\.\hpp") NicStatus
	$(HIPCC) $(CXXFLAGS) $(COMMON_FLAGS) $< -o $@ $(LDFLAGS)

TransferBenchCuda: ./src/client/Client.cpp $(shell find -regex ".*\.\hpp") NicStatus
	$(NVCC) $(NVFLAGS) $(COMMON_FLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f *.o ./TransferBench ./TransferBenchCuda

NicStatus:
  ifeq ($(NIC_ENABLED), 1)
		$(info Building with NIC executor support. Can set DISABLE_NIC_EXEC=1 to disable)
  else
		$(info Building without NIC executor support)
  endif
