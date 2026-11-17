# SPLICE-V: expose the TE0950 block design as an extensible Vitis platform.
#
# TEMPLATE.  Open the TE0950 Vivado project (the reference design of the tool version you
# qualify), open its block design, set the cell/pin names below to the names that exist in
# YOUR design, then source this file in the Vivado Tcl console:
#     source <repo>/platform/te0950/pfm_properties.tcl
# Every 'get_bd_*' below fails loudly if a name does not exist; fix the name, do not guess.
# Reference for the property syntax: AMD XD324 (Versal custom thin platform) and UG1393.

# ---- names to edit -----------------------------------------------------------------
set PFM_NAME        "trenz:te0950:te0950_splicev:1.0"
set CLK_CELL        "/clk_wizard_0"          ;# clocking wizard that drives the kernel clocks
set CLK_PIN         "clk_out1"               ;# 300 MHz output used by the SPLICE-V kernels
set CLK_FREQ_HZ     300000000
set CLK_RESET_CELL  "/psr_300mhz"            ;# proc_sys_reset synchronous to CLK_PIN
set CTRL_SMC_CELL   "/axi_smc_ctrl"          ;# SmartConnect / interconnect for s_axilite control
set CTRL_FIRST_M    1                        ;# first free master index on that interconnect
set CTRL_COUNT      15                       ;# masters the linker may use for kernel control
set NOC_CELL        "/axi_noc_kernels"       ;# NoC instance with free slave ports for m_axi
set NOC_SPTAG       "DDR"                    ;# memory tag the linker should report (LPDDR/DDR)
set NOC_FIRST_S     0
set NOC_COUNT       16                       ;# NoC slave ports offered to kernels
set IRQ_CELL        "/axi_intc_0"            ;# interrupt controller for kernel completion
set XSA_OUT         "te0950_splicev_hw.xsa"
# -------------------------------------------------------------------------------------

set bd [current_bd_design]
set_property PFM_NAME $PFM_NAME [get_files [current_bd_design].bd]

# kernel clock: exactly one default clock; record its id for 'make link CLOCK_ID=<id>'
set_property PFM.CLOCK [list $CLK_PIN [list id "0" is_default "true" proc_sys_reset $CLK_RESET_CELL status "fixed" freq_hz $CLK_FREQ_HZ]] \
    [get_bd_cells $CLK_CELL]

# control masters for kernel s_axilite ports
set ports {}
for {set i $CTRL_FIRST_M} {$i < [expr {$CTRL_FIRST_M + $CTRL_COUNT}]} {incr i} {
    lappend ports [format "M%02d_AXI" $i] [list memport "M_AXI_GP" sptag "" memory "" is_range "true"]
}
set_property PFM.AXI_PORT $ports [get_bd_cells $CTRL_SMC_CELL]

# memory slaves for kernel m_axi ports (through the NoC to DDR)
set ports {}
for {set i $NOC_FIRST_S} {$i < [expr {$NOC_FIRST_S + $NOC_COUNT}]} {incr i} {
    lappend ports [format "S%02d_AXI" $i] [list memport "S_AXI_NOC" sptag $NOC_SPTAG memory "" is_range "true"]
}
set_property PFM.AXI_PORT $ports [get_bd_cells $NOC_CELL]

# kernel interrupts
set_property PFM.IRQ {intr {id 0 range 31}} [get_bd_cells $IRQ_CELL]

# platform intent: embedded Linux, SD boot
set_property platform.extensible true [current_project]
set_property platform.design_intent.embedded true [current_project]
set_property platform.design_intent.server_managed false [current_project]
set_property platform.design_intent.external_host false [current_project]
set_property platform.design_intent.datacenter false [current_project]
set_property platform.default_output_type "sd_card" [current_project]
set_property platform.full_pdi_file "" [current_project]

validate_bd_design
save_bd_design
generate_target all [get_files [current_bd_design].bd]
# The AIE array must be part of the block design (AI Engine IP) for an AIE platform.
write_hw_platform -force -file $XSA_OUT
puts "SPLICE-V: wrote $XSA_OUT; next: python3 platform/te0950/create_platform.py (see platform/README.md)"
