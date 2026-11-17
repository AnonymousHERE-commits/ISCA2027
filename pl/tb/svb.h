// Reader for SVB1 tensor bundles written by tools/make_vectors.py (host and HLS C-sim only).
#ifndef SVB_H
#define SVB_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace svb {

struct Tensor {
    std::vector<uint32_t> shape;
    std::vector<int64_t> data;
    int64_t at(size_t i) const { return data.at(i); }
    size_t size() const { return data.size(); }
};

class Bundle {
public:
    std::map<std::string, Tensor> t;

    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "svb: cannot open %s\n", path.c_str()); return false; }
        uint32_t hdr[2];
        if (std::fread(hdr, 4, 2, f) != 2 || hdr[0] != 0x31425653u) { std::fclose(f); return false; }
        for (uint32_t k = 0; k < hdr[1]; ++k) {
            uint32_t nlen;
            if (std::fread(&nlen, 4, 1, f) != 1) { std::fclose(f); return false; }
            std::string name(nlen, '\0');
            if (nlen && std::fread(&name[0], 1, nlen, f) != nlen) { std::fclose(f); return false; }
            uint32_t pad = (4 - nlen % 4) % 4;
            char skip[4];
            if (pad && std::fread(skip, 1, pad, f) != pad) { std::fclose(f); return false; }
            uint32_t ndim;
            if (std::fread(&ndim, 4, 1, f) != 1) { std::fclose(f); return false; }
            Tensor tt;
            tt.shape.resize(ndim);
            size_t n = 1;
            for (uint32_t d = 0; d < ndim; ++d) {
                if (std::fread(&tt.shape[d], 4, 1, f) != 1) { std::fclose(f); return false; }
                n *= tt.shape[d];
            }
            tt.data.resize(n);
            if (n && std::fread(tt.data.data(), 8, n, f) != n) { std::fclose(f); return false; }
            t[name] = tt;
        }
        std::fclose(f);
        return true;
    }

    const Tensor& operator[](const std::string& name) const {
        auto it = t.find(name);
        if (it == t.end()) { std::fprintf(stderr, "svb: missing tensor %s\n", name.c_str()); std::exit(3); }
        return it->second;
    }

    int64_t scalar(const std::string& name) const { return (*this)[name].at(0); }
};

inline std::string vectors_dir() {
    const char* env = std::getenv("SPLICEV_VECTORS");
    return env ? std::string(env) : std::string("../../artifacts/references/vectors");
}

}  // namespace svb

#endif
