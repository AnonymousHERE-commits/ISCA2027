"""SPLICE-V: create the TE0950 Vitis platform component from the exported XSA.

TEMPLATE for the Vitis Python API (run with:  vitis -s platform/te0950/create_platform.py).
Edit the paths below.  The API names follow AMD's 2024.x/2025.x tutorials (XD324); check
them against `help(vitis)` of your installation before relying on the result, and record
the Vitis version in platform_report.json.
"""
import os

import vitis  # provided by 'vitis -s'

XSA = os.path.abspath("te0950_splicev_hw.xsa")
WORKSPACE = os.path.abspath("build/platform_ws")
NAME = "te0950_splicev"
# Linux artifacts of the image you boot (PetaLinux/EDF build of the same tool version):
SYSROOT = "/path/to/sysroots/cortexa72-cortexa53-xilinx-linux"
BOOT_DIR = "/path/to/linux/images/boot"        # u-boot, bl31, dtb, ... as your flow requires
SD_DIR = "/path/to/linux/images/sd_dir"        # files copied to the FAT partition (boot.scr, ...)

client = vitis.create_client()
client.set_workspace(path=WORKSPACE)
platform = client.create_platform_component(name=NAME, hw_design=XSA, os="linux", cpu="psv_cortexa72",
                                            domain_name="linux_psv_cortexa72")
domain = platform.get_domain("linux_psv_cortexa72")
domain.update_name(new_name="xrt")
domain.add_boot_dir(path=BOOT_DIR)
domain.set_sd_dir(path=SD_DIR)
domain.set_sysroot(path=SYSROOT)
platform.build()
print("platform:", os.path.join(WORKSPACE, NAME, "export", NAME, NAME + ".xpfm"))
vitis.dispose()
