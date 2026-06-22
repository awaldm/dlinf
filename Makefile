CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wpedantic
EIGEN_INCLUDE ?= /usr/include/eigen3
SUPPORT_INCLUDE := support

BUILD_DIR := build
BENCH_RESULTS_DIR := benchmarks/results
DOCS_IMAGES_DIR := docs/images
DEMO_TARGET := $(BUILD_DIR)/dlinf_demo
TEST_LINEAR_TARGET := $(BUILD_DIR)/test_linear
TEST_CONV_TARGET := $(BUILD_DIR)/test_conv2d
TEST_CONV_BN_TARGET := $(BUILD_DIR)/test_conv_bn
TEST_ELEMENTWISE_TARGET := $(BUILD_DIR)/test_elementwise
TEST_BASICBLOCK_TARGET := $(BUILD_DIR)/test_basicblock
BENCH_KERNELS_TARGET := $(BUILD_DIR)/bench_kernels
KERNEL_BENCH_RESULTS ?= $(BENCH_RESULTS_DIR)/local_laptop_kernel_bench.jsonl
KERNEL_LATENCY_SVG ?= $(DOCS_IMAGES_DIR)/kernel_latency.svg
KERNEL_SPEEDUP_SVG ?= $(DOCS_IMAGES_DIR)/kernel_speedup.svg
BENCH_WARMUP ?= 5
BENCH_ITERATIONS ?= 30

COMMON_SOURCES := src/linear.cpp src/conv2d.cpp src/weight_archive.cpp
DEMO_SOURCES := src/main.cpp $(COMMON_SOURCES)
TEST_LINEAR_SOURCES := tests/test_linear.cpp $(COMMON_SOURCES)
TEST_CONV_SOURCES := tests/test_conv2d.cpp $(COMMON_SOURCES)
TEST_CONV_BN_SOURCES := tests/test_conv_bn.cpp src/batchnorm2d.cpp $(COMMON_SOURCES)
TEST_ELEMENTWISE_SOURCES := tests/test_elementwise.cpp
TEST_BASICBLOCK_SOURCES := tests/test_basicblock.cpp src/batchnorm2d.cpp src/basicblock.cpp $(COMMON_SOURCES)
BENCH_KERNELS_SOURCES := benchmarks/bench_kernels.cpp src/batchnorm2d.cpp src/basicblock.cpp $(COMMON_SOURCES)

TARGETS := $(DEMO_TARGET)
ifneq ($(wildcard tests/test_linear.cpp),)
TARGETS += $(TEST_LINEAR_TARGET)
endif
ifneq ($(wildcard tests/test_conv2d.cpp),)
TARGETS += $(TEST_CONV_TARGET)
endif
ifneq ($(wildcard tests/test_conv_bn.cpp src/batchnorm2d.cpp include/dlinf/layers/batchnorm2d.hpp),)
ifneq ($(words $(wildcard tests/test_conv_bn.cpp src/batchnorm2d.cpp include/dlinf/layers/batchnorm2d.hpp)),3)
else
TARGETS += $(TEST_CONV_BN_TARGET)
endif
endif
ifneq ($(wildcard tests/test_elementwise.cpp),)
TARGETS += $(TEST_ELEMENTWISE_TARGET)
endif
ifneq ($(wildcard tests/test_basicblock.cpp),)
TARGETS += $(TEST_BASICBLOCK_TARGET)
endif

.PHONY: all clean demo test-linear test-conv2d test-conv-bn test-elementwise test-basicblock bench-kernels bench-kernels-save plot-benchmarks

all: $(TARGETS)

demo: $(DEMO_TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(DEMO_TARGET): $(DEMO_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Iinclude -I$(EIGEN_INCLUDE) $(DEMO_SOURCES) -o $(DEMO_TARGET)

$(TEST_LINEAR_TARGET): $(TEST_LINEAR_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Iinclude -I$(SUPPORT_INCLUDE) -I$(EIGEN_INCLUDE) $(TEST_LINEAR_SOURCES) -o $(TEST_LINEAR_TARGET)

$(TEST_CONV_TARGET): $(TEST_CONV_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Iinclude -I$(SUPPORT_INCLUDE) -I$(EIGEN_INCLUDE) $(TEST_CONV_SOURCES) -o $(TEST_CONV_TARGET)

$(TEST_CONV_BN_TARGET): $(TEST_CONV_BN_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Iinclude -I$(SUPPORT_INCLUDE) -I$(EIGEN_INCLUDE) $(TEST_CONV_BN_SOURCES) -o $(TEST_CONV_BN_TARGET)

$(TEST_ELEMENTWISE_TARGET): $(TEST_ELEMENTWISE_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Iinclude -I$(EIGEN_INCLUDE) $(TEST_ELEMENTWISE_SOURCES) -o $(TEST_ELEMENTWISE_TARGET)

$(TEST_BASICBLOCK_TARGET): $(TEST_BASICBLOCK_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Iinclude -I$(SUPPORT_INCLUDE) -I$(EIGEN_INCLUDE) $(TEST_BASICBLOCK_SOURCES) -o $(TEST_BASICBLOCK_TARGET)

$(BENCH_KERNELS_TARGET): $(BENCH_KERNELS_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DDLINF_CXXFLAGS='"$(CXXFLAGS)"' -Iinclude -I$(SUPPORT_INCLUDE) -I$(EIGEN_INCLUDE) $(BENCH_KERNELS_SOURCES) -o $(BENCH_KERNELS_TARGET)

test-linear: $(TEST_LINEAR_TARGET)
	$(TEST_LINEAR_TARGET) \
		artifacts/resnet18/resnet18_imagenet1k_v1.elw \
		artifacts/resnet18/fc_golden.elw

test-conv2d: $(TEST_CONV_TARGET)
	$(TEST_CONV_TARGET) \
		artifacts/resnet18/resnet18_imagenet1k_v1.elw \
		artifacts/resnet18/conv1_golden.elw

test-conv-bn:
ifeq ($(words $(wildcard tests/test_conv_bn.cpp src/batchnorm2d.cpp include/dlinf/layers/batchnorm2d.hpp)),3)
	$(MAKE) $(TEST_CONV_BN_TARGET)
	$(TEST_CONV_BN_TARGET) \
		artifacts/resnet18/resnet18_imagenet1k_v1.elw \
		artifacts/resnet18/conv1_bn1_golden.elw
else
	@echo "BatchNorm2D test scaffold exists, but src/batchnorm2d.cpp and include/dlinf/layers/batchnorm2d.hpp are not both present yet."
	@echo "Expected API: dlinf::batchnorm2d(input, weight, bias, running_mean, running_var, eps)"
	@false
endif

test-elementwise: $(TEST_ELEMENTWISE_TARGET)
	$(TEST_ELEMENTWISE_TARGET)

test-basicblock: $(TEST_BASICBLOCK_TARGET)
	$(TEST_BASICBLOCK_TARGET) \
		artifacts/resnet18/resnet18_imagenet1k_v1.elw \
		artifacts/resnet18/layer1_0_basicblock_golden.elw

bench-kernels: $(BENCH_KERNELS_TARGET)
	$(BENCH_KERNELS_TARGET)

bench-kernels-save: $(BENCH_KERNELS_TARGET)
	mkdir -p $(BENCH_RESULTS_DIR)
	$(BENCH_KERNELS_TARGET) --warmup $(BENCH_WARMUP) --iterations $(BENCH_ITERATIONS) > $(KERNEL_BENCH_RESULTS)

plot-benchmarks: $(KERNEL_BENCH_RESULTS)
	mkdir -p $(DOCS_IMAGES_DIR)
	python tools/plot_benchmarks.py \
		--input $(KERNEL_BENCH_RESULTS) \
		--latency-output $(KERNEL_LATENCY_SVG) \
		--speedup-output $(KERNEL_SPEEDUP_SVG)

clean:
	rm -rf $(BUILD_DIR)
