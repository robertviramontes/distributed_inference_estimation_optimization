#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# FORMAT:
# < PARAM TYPE=PARAM_TYPE NAME=PARAM_NAME >
# ARG1_NAME ARG1_PATH_VAL
# ARG2_NAME ARG2_PATH_VAL
# ...
# This starts a section of PARAM definitions, in which each line
# has the syntax below:
# ARG_NAME ARG_PATH_VAL
# ARG_NAME is a macro name for argument value ARG_PATH_VAL.
# PARAM_TYPE can be FILE, or CLOCK.
#
# < POWER_MODEL ID=id_num NAME=mode_name >
# PARAM1_NAME ARG11_NAME ARG11_VAL
# PARAM1_NAME ARG12_NAME ARG12_VAL
# PARAM2_NAME ARG21_NAME ARG21_VAL
# ...
# This starts a section of POWER_MODEL configurations, followed by
# lines with parameter settings as the format below:
# PARAM_NAME ARG_NAME ARG_VAL
# PARAM_NAME and ARG_NAME are defined in PARAM definition sections.
# ARG_VAL is an integer for PARAM_TYPE of CLOCK, and -1 is taken
# as INT_MAX. ARG_VAL is a string for PARAM_TYPE of FILE.
# This file must contain at least one POWER_MODEL section.
#
# < PM_CONFIG DEFAULT=default_mode >
# This is a mandatory section to specify one of the defined power
# model as the default.
import subprocess

PARAM_DEFINITIONS = """
< PARAM TYPE=FILE NAME=CPU_ONLINE >
CORE_0 /sys/devices/system/cpu/cpu0/online
CORE_1 /sys/devices/system/cpu/cpu1/online
CORE_2 /sys/devices/system/cpu/cpu2/online
CORE_3 /sys/devices/system/cpu/cpu3/online

< PARAM TYPE=FILE NAME=GPU_POWER_CONTROL_ENABLE >
GPU_PWR_CNTL_EN /sys/devices/gpu.0/power/control

< PARAM TYPE=FILE NAME=GPU_POWER_CONTROL_DISABLE >
GPU_PWR_CNTL_DIS /sys/devices/gpu.0/power/control

< PARAM TYPE=CLOCK NAME=CPU_A57 >
FREQ_TABLE /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
FREQ_TABLE_KNEXT /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
MAX_FREQ_KNEXT /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
MIN_FREQ_KNEXT /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=GPU >
FREQ_TABLE /sys/devices/gpu.0/devfreq/57000000.gpu/available_frequencies
MAX_FREQ /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq
MIN_FREQ /sys/devices/gpu.0/devfreq/57000000.gpu/min_freq
FREQ_TABLE_KNEXT /sys/devices/17000000.gv11b/devfreq/devfreq0/available_frequencies
MAX_FREQ_KNEXT /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq
MIN_FREQ_KNEXT /sys/devices/gpu.0/devfreq/57000000.gpu/min_freq

< PARAM TYPE=CLOCK NAME=EMC >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/emc_iso_cap
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/emc_iso_cap

###########################
#                         #
# POWER_MODEL DEFINITIONS #
#                         #
###########################\n\n"""


def generate_power_model_definition(gpu_freq, id=0):
    pm_definition = f"<POWER_MODEL ID={id} NAME=EXPERIMENT >\n"
    pm_definition += """
CPU_ONLINE CORE_0 1
CPU_ONLINE CORE_1 1
CPU_ONLINE CORE_2 1
CPU_ONLINE CORE_3 1
CPU_A57 MIN_FREQ  0
CPU_A57 MAX_FREQ -1
GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on\n"""
    pm_definition += f"GPU MIN_FREQ {gpu_freq}\n"
    pm_definition += f"GPU MAX_FREQ {gpu_freq}\n"
    pm_definition += """GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto
EMC MAX_FREQ 0

< PM_CONFIG DEFAULT=0 >
    """

    return pm_definition


def restore():
    subprocess.run(["nvpmodel", "-m", "0", "-f", "/etc/nvpmodel.conf"])
