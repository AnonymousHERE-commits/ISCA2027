// Host conformance test of the AIE dispatcher bodies (scalar path) against the integer
// reference: replays artifacts/references/vectors/units/<suite>.cmd.bin through the same
// dispatch_one() the AIE tile runs and compares every response word with <suite>.rsp.bin.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define SVGEN_AIE 1   // same view of the generated header as the AIE tiles
#include "sv_units.h"

struct VecIO {
    const std::vector<uint32_t>& in;
    size_t pos;
    std::vector<uint32_t> out;
    explicit VecIO(const std::vector<uint32_t>& v) : in(v), pos(0) {}
    uint32_t read() { return pos < in.size() ? in[pos++] : 0u; }
    void write(uint32_t w) { out.push_back(w); }
};

static std::vector<uint32_t> load(const std::string& path) {
    std::vector<uint32_t> v;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return v;
    uint32_t w;
    while (std::fread(&w, 4, 1, f) == 1) v.push_back(w);
    std::fclose(f);
    return v;
}

static uint32_t g_scratch[svgen::SCRATCH_BYTES / 4];

int main(int argc, char** argv) {
    const char* env = std::getenv("SPLICEV_VECTORS");
    std::string dir = std::string(env ? env : "../../artifacts/references/vectors") + "/units/";
    std::vector<std::string> suites;
    for (int i = 1; i < argc; ++i) suites.push_back(argv[i]);
    if (suites.empty()) suites = {"basic", "toy0", "toy1"};
    int bad = 0;
    for (const auto& s : suites) {
        std::vector<uint32_t> cmd = load(dir + s + ".cmd.bin"), rsp = load(dir + s + ".rsp.bin");
        if (cmd.empty() || rsp.empty()) { std::printf("FAIL: missing vectors for %s (run python3 tools/make_vectors.py)\n", s.c_str()); return 2; }
        VecIO io(cmd);
        int units = 0;
        while (io.pos < cmd.size()) { sva::dispatch_one(io, g_scratch); ++units; }
        size_t n = io.out.size() < rsp.size() ? io.out.size() : rsp.size();
        size_t mism = 0, first = (size_t)-1;
        for (size_t i = 0; i < n; ++i)
            if (io.out[i] != rsp[i]) { ++mism; if (first == (size_t)-1) first = i; }
        bool ok = mism == 0 && io.out.size() == rsp.size();
        std::printf("%-6s %3d units, %6zu response words: %s", s.c_str(), units, rsp.size(), ok ? "bit-exact\n" : "MISMATCH");
        if (!ok) std::printf(" (%zu words differ, first at %zu, got %zu words)\n", mism, first, io.out.size());
        bad |= !ok;
    }
    std::printf(bad ? "FAIL\n" : "PASS: dispatcher bodies are bit-exact against the integer reference\n");
    return bad;
}
