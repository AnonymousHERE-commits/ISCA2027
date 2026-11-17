// sv_mm2s / sv_s2mm: DDR <-> AIE PLIO movers (64-bit AXI4-Stream).  Used by the G02 smoke
// and, one pair per worker, to feed dispatcher units in the host-orchestrated bring-up.
// The PS arms s2mm before starting mm2s (roadmap 11.2).  Lengths count 64-bit beats.
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

typedef ap_axiu<64, 0, 0, 0> beat_t;

extern "C" void sv_mm2s(const ap_uint<64>* mem, hls::stream<beat_t>& s, unsigned int n_beats) {
#pragma HLS INTERFACE m_axi port = mem offset = slave bundle = gmem depth = 65536 max_read_burst_length = 256
#pragma HLS INTERFACE axis port = s
#pragma HLS INTERFACE s_axilite port = n_beats
#pragma HLS INTERFACE s_axilite port = return
    for (unsigned int i = 0; i < n_beats; ++i) {
#pragma HLS PIPELINE II = 1
        beat_t v;
        v.data = mem[i];
        v.keep = -1;
        v.strb = -1;
        v.last = (i == n_beats - 1) ? 1 : 0;
        s.write(v);
    }
}

extern "C" void sv_s2mm(ap_uint<64>* mem, hls::stream<beat_t>& s, unsigned int n_beats) {
#pragma HLS INTERFACE m_axi port = mem offset = slave bundle = gmem depth = 65536 max_write_burst_length = 256
#pragma HLS INTERFACE axis port = s
#pragma HLS INTERFACE s_axilite port = n_beats
#pragma HLS INTERFACE s_axilite port = return
    for (unsigned int i = 0; i < n_beats; ++i) {
#pragma HLS PIPELINE II = 1
        beat_t v = s.read();
        mem[i] = v.data;
    }
}
