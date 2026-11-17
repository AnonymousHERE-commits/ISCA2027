// Native AIE graph control for the Python host when the installed pyxrt has no graph
// binding (roadmap 3.3: 'an explicitly documented binding layer, not an invented API call').
// Build on the board: make -C ps/native   (needs the XRT headers and libxrt_coreutil).
//
// XRT semantics used here: graph.run(0) runs until the graph is ended; graph.end(cycles)
// stops it 'cycles' AIE cycles after the last run (0 would wait for completion, which a
// persistent dispatcher graph never reaches).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>

#include "xrt/xrt_device.h"
#include "xrt/xrt_graph.h"
#include "xrt/xrt_uuid.h"

struct SvGraph {
    xrt::device dev;
    xrt::uuid uuid;
    xrt::graph graph;
};

extern "C" void* sv_graph_open(const char* xclbin, const char* name) {
    try {
        std::unique_ptr<SvGraph> g(new SvGraph());
        g->dev = xrt::device(0);
        g->uuid = g->dev.load_xclbin(std::string(xclbin));
        g->graph = xrt::graph(g->dev, g->uuid, std::string(name));
        return g.release();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sv_graph_open: %s\n", e.what());
        return nullptr;
    }
}

extern "C" int sv_graph_run(void* h, int iterations) {
    try {
        static_cast<SvGraph*>(h)->graph.run(static_cast<uint32_t>(iterations < 0 ? 0 : iterations));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sv_graph_run: %s\n", e.what());
        return 1;
    }
}

extern "C" int sv_graph_wait(void* h, int timeout_ms) {
    try {
        static_cast<SvGraph*>(h)->graph.wait(std::chrono::milliseconds(timeout_ms));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sv_graph_wait: %s\n", e.what());
        return 1;
    }
}

extern "C" int sv_graph_end(void* h, unsigned long long cycles) {
    try {
        static_cast<SvGraph*>(h)->graph.end(static_cast<uint64_t>(cycles));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sv_graph_end: %s\n", e.what());
        return 1;
    }
}

extern "C" void sv_graph_close(void* h) {
    delete static_cast<SvGraph*>(h);
}
