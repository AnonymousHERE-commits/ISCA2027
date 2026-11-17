// Host run of the G02 smoke graph with the mock ADF layer (see host_graph.cpp).
//   usage: host_smoke <data_dir> <iterations>   reads smoke_in.txt, writes smoke_out.txt
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "smoke_graph.h"

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    const std::string dir = argv[1];
    input_stream<int32> in;
    output_stream<int32> out;
    std::ifstream f(dir + "/smoke_in.txt");
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        long long a;
        while (ss >> a) in.q.push_back((int32)a);
    }
    SmokeGraph g;
    for (int i = 0; i < std::atoi(argv[2]); ++i) g.k.fn(&in, &out);
    std::ofstream o(dir + "/smoke_out.txt");
    for (size_t w = 0; w + 1 < out.v.size(); w += 2) o << out.v[w] << " " << out.v[w + 1] << "\n";
    std::printf("smoke: %zu words out, %zu input words left\n", out.v.size(), in.q.size());
    return in.q.empty() ? 0 : 1;
}
