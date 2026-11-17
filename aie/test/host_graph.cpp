// Host run of a generated SPLICE-V worker graph (tools/gen_aie_graph.py) with the mock ADF
// layer: constructs the graph exactly as the AIE compiler would see it, then runs each
// worker's kernel SV_SIM_UNITS times on data/w<i>_in.txt and writes data/w<i>_out.txt in
// the simulator's text format.  tools/aie_sim_data.py check compares the outputs with the
// integer reference routed by the same worker classes.
//   usage: host_graph <data_dir> <units_per_worker>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "sv_graph.h"

static bool load_plio(const std::string& path, input_stream<int32>& s) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == 'T') continue;
        std::istringstream ss(line);
        long long a;
        while (ss >> a) s.q.push_back((int32)a);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <data_dir> <units_per_worker>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const int units = std::atoi(argv[2]);
    SvWorkersGraph graph;
    graph.init();
    for (int i = 0; i < SV_N_WORKERS; ++i) {
        input_stream<int32> in;
        output_stream<int32> out;
        const std::string base = dir + "/w" + std::to_string(i);
        if (!load_plio(base + "_in.txt", in)) {
            std::fprintf(stderr, "missing %s_in.txt\n", base.c_str());
            return 2;
        }
        try {
            for (int u = 0; u < units; ++u) graph.k[i].fn(&in, &out);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "worker %d (%s): %s\n", i, graph.k[i].src.c_str(), e.what());
            return 1;
        }
        if (!in.q.empty()) std::fprintf(stderr, "worker %d: %zu input words left unread\n", i, in.q.size());
        std::ofstream f(base + "_out.txt");
        for (size_t w = 0; w + 1 < out.v.size(); w += 2) f << out.v[w] << " " << out.v[w + 1] << "\n";
        if (out.v.size() & 1) f << out.v.back() << "\n";
        std::printf("worker %d %-16s %d units, %zu words out\n", i, graph.k[i].src.c_str(), units, out.v.size());
    }
    graph.end();
    return 0;
}
