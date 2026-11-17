// G02 smoke graph: one AIE-ML kernel between one 64-bit input PLIO and one output PLIO.
#ifndef SMOKE_GRAPH_H
#define SMOKE_GRAPH_H

#include <adf.h>

#include "sv_smoke.h"

class SmokeGraph : public adf::graph {
public:
    adf::kernel k;
    adf::input_plio in;
    adf::output_plio out;
    SmokeGraph() {
        k = adf::kernel::create(sv_smoke);
        adf::source(k) = "sv_smoke.cc";
        adf::runtime<adf::ratio>(k) = 0.9;
        in = adf::input_plio::create("smoke_in", adf::plio_64_bits, "data/smoke_in.txt");
        out = adf::output_plio::create("smoke_out", adf::plio_64_bits, "data/smoke_out.txt");
        adf::connect(in.out[0], k.in[0]);
        adf::connect(k.out[0], out.in[0]);
    }
};

#endif
