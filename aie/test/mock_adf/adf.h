// Host mock of the small ADF subset used by the SPLICE-V worker graph and kernels.
// It lets g++ compile the generated graph and worker sources unchanged and run each
// worker's kernel on PLIO text files (aie/test/host_graph.cpp): a syntax and protocol
// check before x86simulator/aiesimulator.  It models nothing about timing, placement,
// memory or the AIE compiler; only the real tools can qualify those (gate G05).
#ifndef SV_MOCK_ADF_H
#define SV_MOCK_ADF_H

#include <cstdint>
#include <cstdio>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

typedef int32_t int32;

template <typename T>
struct input_stream {
    std::deque<T> q;
};
template <typename T>
struct output_stream {
    std::vector<T> v;
};

template <typename T>
static inline T readincr(input_stream<T>* s) {
    if (s->q.empty()) throw std::runtime_error("mock adf: kernel read past the end of its input stream");
    T x = s->q.front();
    s->q.pop_front();
    return x;
}
template <typename T>
static inline void writeincr(output_stream<T>* s, T x) { s->v.push_back(x); }

namespace adf {

enum plio_type { plio_32_bits = 32, plio_64_bits = 64, plio_128_bits = 128 };
struct ratio {};

typedef void (*stream_kernel_fn)(input_stream<int32>*, output_stream<int32>*);

struct port {
    std::string owner;
    int index;
};

struct kernel {
    stream_kernel_fn fn = nullptr;
    std::string src;
    double ratio_v = 0.0;
    int stack = 0;
    port in[1];
    port out[1];
    static kernel create(stream_kernel_fn f) {
        kernel k;
        k.fn = f;
        return k;
    }
};

struct input_plio {
    std::string name, file;
    port out[1];
    static input_plio create(const std::string& n, plio_type w, const std::string& f, double = 0.0) {
        if (w != plio_64_bits) throw std::runtime_error("mock adf: SPLICE-V uses 64-bit PLIOs");
        input_plio p;
        p.name = n;
        p.file = f;
        return p;
    }
};
struct output_plio {
    std::string name, file;
    port in[1];
    static output_plio create(const std::string& n, plio_type w, const std::string& f, double = 0.0) {
        if (w != plio_64_bits) throw std::runtime_error("mock adf: SPLICE-V uses 64-bit PLIOs");
        output_plio p;
        p.name = n;
        p.file = f;
        return p;
    }
};

static inline std::string& source(kernel& k) { return k.src; }
template <typename R>
static inline double& runtime(kernel& k) { return k.ratio_v; }
static inline int& stack_size(kernel& k) { return k.stack; }
static inline void connect(const port&, const port&) {}

struct graph {
    int init() { return 0; }
    int run(int = -1) { return 0; }
    int end() { return 0; }
};

}  // namespace adf

#endif
