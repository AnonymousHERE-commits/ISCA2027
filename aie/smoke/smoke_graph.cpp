#include "smoke_graph.h"

#if defined(__AIESIM__) || defined(__X86SIM__)
#include <cstdlib>
#endif

SmokeGraph smoke;

#if defined(__AIESIM__) || defined(__X86SIM__)
#ifndef SMOKE_SIM_ITER
#define SMOKE_SIM_ITER 3
#endif
int main() {
    const char* env = std::getenv("SMOKE_SIM_ITER");
    smoke.init();
    smoke.run(env ? std::atoi(env) : SMOKE_SIM_ITER);
    smoke.end();
    return 0;
}
#endif
