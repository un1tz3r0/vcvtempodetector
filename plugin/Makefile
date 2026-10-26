RACK_DIR ?= ../..

FLAGS += -Isrc
SOURCES += $(wildcard src/*.cpp src/dsp/*.cpp)
DISTRIBUTABLES += res

# The offline driver for python/beatref (`beatref compare`, `--engine cpp`) needs no Rack SDK.
ifneq ($(MAKECMDGOALS),beattrack)
include $(RACK_DIR)/plugin.mk
CXXFLAGS := $(filter-out -std=c++11,$(CXXFLAGS)) -std=c++17
# Self-contained DLL: don't depend on Rack shipping a matching libgcc_s_seh / libwinpthread.
ifdef ARCH_WIN
LDFLAGS += -static-libgcc -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive,-Bdynamic
endif
endif

build/beattrack: tools/beattrack.cpp $(wildcard src/dsp/*.cpp src/dsp/*.hpp)
	@mkdir -p build
	$(CXX) -O3 -funsafe-math-optimizations -std=c++17 -Wall -o $@ tools/beattrack.cpp $(wildcard src/dsp/*.cpp)

beattrack: build/beattrack
.PHONY: beattrack
