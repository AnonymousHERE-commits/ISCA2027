# Builds libsv_graph.so on the board (or with a cross toolchain and the platform sysroot).
XILINX_XRT ?= /usr
CXX ?= g++
libsv_graph.so: sv_graph_ctl.cpp
	$(CXX) -std=c++17 -O2 -shared -fPIC $< -I$(XILINX_XRT)/include -L$(XILINX_XRT)/lib -lxrt_coreutil -o $@
clean:
	rm -f libsv_graph.so
.PHONY: clean
