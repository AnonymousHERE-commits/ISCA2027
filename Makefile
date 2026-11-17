# SPLICE-V build and verification entry points.  `make help` lists the targets.
#
# Host-only targets (no AMD tools needed): test generate check-generated contract audit
#   vectors host-cpp emulate.
# Tool targets (source the Vitis 2025.2 settings first): hls-csim hls hls-cosim aie aie-sim
#   link package release system bundle.  PLATFORM must name the qualified TE0950 .xpfm for
#   every hardware target; nothing here invents a platform (roadmap 3, 11.3).
#
# Every recipe runs under bash with -e -o pipefail, so a failing tool fails the target even
# when its output is piped into a log (roadmap 11.4: a log file is not a successful build).

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

REPO      := $(abspath .)
PY        ?= python3
B         := $(REPO)/build
LOGS      := $(REPO)/logs
VECTORS   ?= $(REPO)/artifacts/references/vectors
CXX       ?= g++
HOSTFLAGS := -std=c++17 -O2 -Wall -Wextra -Wno-unknown-pragmas

PLATFORM      ?=
SYSTEM        ?= slice1
KERNEL        ?= sv_gate
SUITE         ?= basic
AIE_TARGET    ?= hw
AIE_PL_FREQ   ?= 300
AIE_AUTOSTART ?= 0
AIE_FLAGS     ?=
VPP_LINK_FLAGS ?=
VPP_PKG_FLAGS ?=
CLOCK_ID      ?=
ROOTFS        ?=
KERNEL_IMAGE  ?=

# system table (pl/kernels.json): AIE graph set, kernel list and capacity set of SYSTEM
SYS_AIE   = $(shell $(PY) -c "import json;print(json.load(open('pl/kernels.json'))['systems']['$(SYSTEM)']['aie'])")
SYS_KERNS = $(shell $(PY) -c "import json;print(' '.join(json.load(open('pl/kernels.json'))['systems']['$(SYSTEM)']['kernels']))")
CAPS      ?= $(shell $(PY) -c "import json;print(json.load(open('pl/kernels.json'))['systems']['$(SYSTEM)']['caps'])")
PART      ?= $(shell $(PY) -c "import json;print(json.load(open('pl/kernels.json'))['part'])")
AIE_SET   ?= $(SYS_AIE)

HLS_DIR   = $(B)/hls/$(CAPS)/$(KERNEL)
AIE_SRC   = $(B)/aie/$(AIE_SET)/src
AIE_DIR   = $(B)/aie/$(AIE_SET)/$(AIE_TARGET)
SYS_DIR   = $(B)/system/$(SYSTEM)
LIBADF    = $(B)/aie/$(SYS_AIE)/hw/libadf.a
XOS       = $(foreach k,$(SYS_KERNS),$(B)/hls/$(CAPS)/$(k)/$(k).xo)

SIMULATOR = $(if $(filter x86sim,$(AIE_TARGET)),x86simulator,aiesimulator)
SIM_OUT   = $(if $(filter x86sim,$(AIE_TARGET)),x86simulator_output,aiesimulator_output)

need_platform = @if [ -z "$(PLATFORM)" ] || [ ! -f "$(PLATFORM)" ]; then echo "PLATFORM must name the qualified TE0950 .xpfm (got '$(PLATFORM)'); see platform/README.md"; exit 2; fi
need_tool = @command -v $(1) >/dev/null || { echo "$(1) not found: source the Vitis 2025.2 settings64.sh first"; exit 2; }

.PHONY: help test generate check-generated contract audit vectors host-cpp host-pl host-aie host-graphs emulate \
        preflight-host hls-cfg hls-csim hls hls-cosim hls-system aie-src aie aie-sim link package \
        release system bundle native clean

help:  ## list the targets
	@grep -E '^[a-z][a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/' | expand -t 18

# ---------------------------------------------------------------------- host only
test: | $(LOGS)  ## all host tests (Python reference, generated sources, PS paths on the emulated device)
	$(PY) -m unittest discover -s tests -v 2>&1 | tee $(LOGS)/test.log

generate:  ## regenerate generated/ and detector/export_spec.json from contracts/
	$(PY) tools/generate_sources.py
	$(PY) tools/make_export_spec.py

check-generated:  ## fail if generated/ or detector/export_spec.json is stale with respect to the manifest
	$(PY) tools/generate_sources.py --check
	$(PY) tools/make_export_spec.py --check

contract:  ## list unresolved (null), synthetic (SYN) and design-decision (DD) manifest fields
	$(PY) tools/check_contract.py

audit:  ## recompute the paper's numbers (evidence/audit/paper_numbers.json)
	$(PY) tools/audit_paper_numbers.py

vectors:  ## bit-exact test vectors from the integer reference
	$(PY) tools/make_vectors.py --out $(VECTORS) > /dev/null
	@echo "vectors -> $(VECTORS)"

host-cpp: host-pl host-aie host-graphs  ## C++ engines, dispatcher and generated graphs vs the reference (g++)

host-pl: vectors | $(LOGS)  ## PL cores and kernel tops (g++, __int128) vs the reference
	@mkdir -p $(B)/host
	$(CXX) $(HOSTFLAGS) -Ipl/include -Ipl/engines -Igenerated/cpp -Ipl/tb pl/tb/tb_cores.cpp -o $(B)/host/tb_cores
	$(CXX) $(HOSTFLAGS) -DTB_ALL -Ipl/include -Ipl/engines -Igenerated/cpp -Ipl/tb pl/tb/tb_kernels.cpp \
	    pl/engines/sv_reserve_k.cpp pl/engines/sv_linkage_k.cpp pl/engines/sv_gate_k.cpp pl/engines/sv_exchange_k.cpp -o $(B)/host/tb_kernels
	SPLICEV_VECTORS=$(VECTORS) $(B)/host/tb_cores 2>&1 | tee $(LOGS)/host_tb_cores.log
	SPLICEV_VECTORS=$(VECTORS) $(B)/host/tb_kernels 2>&1 | tee $(LOGS)/host_tb_kernels.log

host-aie: vectors | $(LOGS)  ## AIE dispatcher bodies (scalar path, g++) vs the reference
	@mkdir -p $(B)/host
	$(CXX) $(HOSTFLAGS) -Iaie/kernels -Iaie/include -Igenerated/cpp aie/test/host_units.cpp -o $(B)/host/host_units
	SPLICEV_VECTORS=$(VECTORS) $(B)/host/host_units 2>&1 | tee $(LOGS)/host_units.log

host-graphs: vectors | $(LOGS)  ## generated worker graphs (mock ADF) through the simulator data path, every class set
	@set -e; for S in uniform1 split3 split6; do \
	  D=$(B)/host/graph_$$S; rm -rf $$D; \
	  $(PY) tools/gen_aie_graph.py --set $$S --out $$D/src > /dev/null; \
	  $(CXX) $(HOSTFLAGS) -Wno-unused-function -Iaie/test/mock_adf -I$$D/src -Iaie/kernels -Iaie/include -Igenerated/cpp \
	      aie/test/host_graph.cpp $$D/src/sv_worker_m*.cc -o $$D/host_graph; \
	  for T in basic toy0 toy1; do \
	    $(PY) tools/aie_sim_data.py make --suite $$T --vectors $(VECTORS) --workers-json $$D/src/workers.json --out $$D/data_$$T > /dev/null; \
	    U=$$($(PY) -c "import json;print(json.load(open('$$D/data_$$T/units.json'))['SV_SIM_UNITS'])"); \
	    mkdir -p $$D/out_$$T; cp $$D/data_$$T/w*_in.txt $$D/out_$$T/; \
	    $$D/host_graph $$D/out_$$T $$U > /dev/null; \
	    echo "$$S/$$T: $$($(PY) tools/aie_sim_data.py check --sim $$D/out_$$T --expected $$D/data_$$T | tail -1)"; \
	    $(PY) tools/aie_sim_data.py check --sim $$D/out_$$T --expected $$D/data_$$T > /dev/null; \
	  done; \
	done 2>&1 | tee $(LOGS)/host_graphs.log
	@D=$(B)/host/smoke; rm -rf $$D; mkdir -p $$D; $(PY) tools/aie_sim_data.py smoke --out $$D > /dev/null; \
	  $(CXX) $(HOSTFLAGS) -Iaie/test/mock_adf -Iaie/smoke aie/test/host_smoke.cpp aie/smoke/sv_smoke.cc -o $$D/host_smoke; \
	  $$D/host_smoke $$D 3 > /dev/null; echo "smoke: $$($(PY) tools/aie_sim_data.py check --sim $$D --expected $$D | tail -1)"; \
	  $(PY) tools/aie_sim_data.py check --sim $$D --expected $$D > /dev/null

emulate: vectors  ## the board applications on the emulated device (checks the host program, not the board)
	$(PY) ps/apps/g02_smoke.py --backend emulated
	$(PY) ps/apps/g05_units.py --backend emulated --vectors $(VECTORS) --masks 0x03,0x44,0xB8
	$(PY) ps/apps/g06_engines.py --backend emulated --vectors $(VECTORS)
	$(PY) ps/apps/g07_slice.py --backend emulated

preflight-host:  ## read-only inventory of this machine (G01) -> evidence/preflight/host.json
	$(PY) tools/preflight.py --target host --output evidence/preflight/host.json $(if $(PLATFORM),--xpfm $(PLATFORM))

$(LOGS):
	@mkdir -p $(LOGS)

# ---------------------------------------------------------------------- PL (Vitis HLS)
hls-cfg:
	$(PY) tools/build_cfg.py hls --kernel $(KERNEL) --caps $(CAPS) --out $(HLS_DIR)/hls_config.cfg > /dev/null

hls-csim: hls-cfg vectors | $(LOGS)  ## C simulation of KERNEL (CAPS=small|full)
	$(call need_tool,vitis-run)
	cd $(HLS_DIR) && SPLICEV_VECTORS=$(VECTORS) vitis-run --mode hls --csim --config hls_config.cfg --work_dir work \
	    2>&1 | tee $(LOGS)/hls_csim_$(CAPS)_$(KERNEL).log
	@grep -q -E "PASS" $(LOGS)/hls_csim_$(CAPS)_$(KERNEL).log || { echo "csim log has no PASS line"; exit 1; }

hls: hls-cfg | $(LOGS)  ## synthesize and package KERNEL as .xo
	$(call need_tool,v++)
	cd $(HLS_DIR) && v++ -c --mode hls --config hls_config.cfg --work_dir work 2>&1 | tee $(LOGS)/hls_$(CAPS)_$(KERNEL).log
	@test -f $(HLS_DIR)/$(KERNEL).xo || { echo "no $(HLS_DIR)/$(KERNEL).xo was produced"; exit 1; }

hls-cosim: hls-cfg vectors | $(LOGS)  ## RTL co-simulation of KERNEL (after make hls)
	$(call need_tool,vitis-run)
	cd $(HLS_DIR) && SPLICEV_VECTORS=$(VECTORS) vitis-run --mode hls --cosim --config hls_config.cfg --work_dir work \
	    2>&1 | tee $(LOGS)/hls_cosim_$(CAPS)_$(KERNEL).log

hls-system:  ## every kernel of SYSTEM as .xo
	@for k in $(SYS_KERNS); do $(MAKE) --no-print-directory hls KERNEL=$$k CAPS=$(CAPS); done

# ---------------------------------------------------------------------- AIE
aie-src:
	@if [ "$(AIE_SET)" != "smoke" ]; then $(PY) tools/gen_aie_graph.py --set $(AIE_SET) --out $(AIE_SRC) > /dev/null; fi

aie: aie-src | $(LOGS)  ## compile the AIE graph AIE_SET (AIE_TARGET=hw|x86sim) -> libadf.a
	$(call need_tool,v++)
	@if [ "$(AIE_TARGET)" = "hw" ] && { [ -z "$(PLATFORM)" ] || [ ! -f "$(PLATFORM)" ]; }; then \
	  echo "AIE_TARGET=hw needs PLATFORM=<TE0950 .xpfm>; use AIE_TARGET=x86sim for a simulation-only build"; exit 2; fi
	@mkdir -p $(AIE_DIR)
	cd $(AIE_DIR) && v++ -c --mode aie --target=$(AIE_TARGET) $(if $(PLATFORM),--platform=$(PLATFORM),--part=$(PART)) \
	    $(if $(filter smoke,$(AIE_SET)),--include=$(REPO)/aie/smoke,--include=$(AIE_SRC)) \
	    --include=$(REPO)/aie/kernels --include=$(REPO)/aie/include --include=$(REPO)/generated/cpp \
	    --aie.pl-freq=$(AIE_PL_FREQ) --work_dir=./Work $(AIE_FLAGS) \
	    $(if $(filter smoke,$(AIE_SET)),$(REPO)/aie/smoke/smoke_graph.cpp,$(AIE_SRC)/sv_graph.cpp) \
	    2>&1 | tee $(LOGS)/aie_$(AIE_SET)_$(AIE_TARGET).log
	@test -f $(AIE_DIR)/libadf.a || { echo "no libadf.a in $(AIE_DIR)"; exit 1; }

aie-sim: vectors | $(LOGS)  ## simulate AIE_SET on SUITE (x86simulator if AIE_TARGET=x86sim, else aiesimulator)
	@mkdir -p $(AIE_DIR)
ifeq ($(AIE_SET),smoke)
	$(PY) tools/aie_sim_data.py smoke --out $(AIE_DIR)/data > /dev/null
else
	$(PY) tools/aie_sim_data.py make --suite $(SUITE) --vectors $(VECTORS) --workers-json $(AIE_SRC)/workers.json \
	    --out $(AIE_DIR)/data > /dev/null
endif
	cd $(AIE_DIR) && export SV_SIM_UNITS=$$($(PY) $(REPO)/tools/aie_sim_data.py units --data data) SMOKE_SIM_ITER=3 && \
	    $(SIMULATOR) --pkg-dir=./Work 2>&1 | tee $(LOGS)/aie_sim_$(AIE_SET)_$(AIE_TARGET).log
	$(PY) tools/aie_sim_data.py check --expected $(AIE_DIR)/data --sim $(AIE_DIR)/$(SIM_OUT)/data

# ---------------------------------------------------------------------- system
link: | $(LOGS)  ## v++ --link SYSTEM (AIE archive + kernel objects + PLATFORM) -> splicev.xsa
	$(call need_tool,v++)
	$(call need_platform)
	@mkdir -p $(SYS_DIR)
	$(PY) tools/build_cfg.py connectivity --system $(SYSTEM) --out $(SYS_DIR)/connectivity.cfg \
	    $(if $(filter-out smoke,$(SYS_AIE)),--workers-json $(B)/aie/$(SYS_AIE)/src/workers.json) $(if $(CLOCK_ID),--clock-id $(CLOCK_ID)) > /dev/null
	@for f in $(LIBADF) $(XOS); do test -f $$f || { echo "missing $$f (make aie AIE_SET=$(SYS_AIE) / make hls-system SYSTEM=$(SYSTEM))"; exit 1; }; done
	cd $(SYS_DIR) && v++ --link --target hw --platform $(PLATFORM) --config connectivity.cfg --save-temps --temp_dir ./_x_link \
	    $(VPP_LINK_FLAGS) $(XOS) $(LIBADF) -o splicev.xsa 2>&1 | tee $(LOGS)/link_$(SYSTEM).log
	@test -f $(SYS_DIR)/splicev.xsa

package: | $(LOGS)  ## v++ --package SYSTEM -> splicev.xclbin (+ SD boot files)
	$(call need_tool,v++)
	$(call need_platform)
	$(PY) tools/build_cfg.py package --out $(SYS_DIR)/package.cfg --out-dir $(SYS_DIR)/package \
	    $(if $(filter 1,$(AIE_AUTOSTART)),--autostart) $(if $(ROOTFS),--rootfs $(ROOTFS)) $(if $(KERNEL_IMAGE),--kernel-image $(KERNEL_IMAGE)) > /dev/null
	cd $(SYS_DIR) && v++ --package --target hw --platform $(PLATFORM) --config package.cfg --temp_dir ./_x_package \
	    $(VPP_PKG_FLAGS) splicev.xsa $(LIBADF) -o splicev.xclbin 2>&1 | tee $(LOGS)/package_$(SYSTEM).log
	@test -f $(SYS_DIR)/splicev.xclbin

release:  ## bind SYSTEM into artifacts/release/SYSTEM/release.json + xclbin (refuses an incomplete build)
	$(call need_platform)
	$(PY) tools/release.py --system $(SYSTEM) --platform $(PLATFORM) --caps $(CAPS) $(if $(filter 1,$(AIE_AUTOSTART)),--autostart)

system:  ## hls-system + aie + link + package + release for SYSTEM
	$(MAKE) --no-print-directory hls-system SYSTEM=$(SYSTEM)
	$(MAKE) --no-print-directory aie SYSTEM=$(SYSTEM) AIE_SET=$(SYS_AIE) AIE_TARGET=hw
	$(MAKE) --no-print-directory link SYSTEM=$(SYSTEM)
	$(MAKE) --no-print-directory package SYSTEM=$(SYSTEM)
	$(MAKE) --no-print-directory release SYSTEM=$(SYSTEM)

bundle: vectors  ## tarball of what the board needs for SYSTEM (PS runtime, reference, vectors, release)
	@test -f $(REPO)/artifacts/release/$(SYSTEM)/release.json || { echo "no release for $(SYSTEM): make system SYSTEM=$(SYSTEM)"; exit 1; }
	@mkdir -p $(B)/bundle
	tar -czf $(B)/bundle/splicev_$(SYSTEM).tar.gz --exclude=__pycache__ --exclude='*.so' \
	    ps reference generated/py tools/probe_pyxrt.py tools/preflight.py contracts \
	    artifacts/references/vectors artifacts/release/$(SYSTEM)
	@echo "$(B)/bundle/splicev_$(SYSTEM).tar.gz"

native:  ## (on the board) XRT C++ graph-control helper for pyxrt builds without a graph binding
	$(MAKE) -C ps/native

clean:  ## remove build outputs (keeps logs and evidence)
	rm -rf $(B)/host $(B)/hls $(B)/aie $(B)/system $(B)/bundle
