import numpy as np
from math import floor
import pandas as pd

LEPOTATO_NAME = "lepotato"
NANO_NAME = "nano"
WISCAD_NAME = "wiscad"


def trimmean(vals, p):
    vals = np.sort(np.array(vals).flatten())
    num_skip = floor(p * len(vals) / 2)
    return np.mean(vals[num_skip : (-num_skip or None)])


def potato_trimmean(df_group, p):
    indices = np.argsort(np.array(df_group["durs"]))
    num_skip = floor(p * len(indices) / 2)
    num_skip = num_skip if num_skip % 2 == 0 else num_skip - 1

    indices = [
        idx
        for idx in indices
        if idx > num_skip and idx <= (len(indices) - num_skip)
    ]
    mean_durs = np.mean(df_group["durs"].to_numpy()[indices])
    mean_energy = np.mean(df_group["run_energy"].to_numpy()[indices])
    result = pd.DataFrame(
        {
            "layers": [list(df_group["layers"])[0]],
            "frequency": [list(df_group["frequency"])[0]],
            "durs": [mean_durs],
            "run_energy": [mean_energy],
        }
    )
    return result


def estimate_communication_latency(
    to_layer: str, bandwidth: float, get_input_shape_fn, unit_size=4
) -> float:
    if to_layer == "output":
        intermediate_tensor_size = [1]
    else:
        intermediate_tensor_size = get_input_shape_fn(to_layer)
    intermediate_tensor_size_B = np.prod(intermediate_tensor_size) * unit_size
    return intermediate_tensor_size_B / bandwidth


SPEED_DOWNLOAD = {
    "3G": (2.0275 / 8) * 1e6,
    "4G": (13.76 / 8) * 1e6,
    "WiFi": (54.97 / 8) * 1e6,
}
SPEED_UPLOAD = {
    "3G": (1.1 / 8) * 1e6,
    "4G": (5.85 / 8) * 1e6,
    "WiFi": (18.88 / 8) * 1e6,
}


ALPHA_UPLOAD = {
    "3G": 868.98,
    "4G": 438.39,
    "WiFi": 283.17,
}
ALPHA_DOWNLOAD = {
    "3G": 122.12,
    "4G": 51.97,
    "WiFi": 137.01,
}
BETA = {
    "3G": 817.88,
    "4G": 1288.04,
    "WiFi": 132.86,
}


def estimate_communication_latency_asymmetric(
    to_layer: str,
    conn_name: str,
    src_name: str,
    get_input_shape_fn,
    unit_size=4,
):
    if to_layer == "output":
        intermediate_tensor_size = [1]
    else:
        intermediate_tensor_size = get_input_shape_fn(to_layer)
    intermediate_tensor_size_B = np.prod(intermediate_tensor_size) * unit_size

    if src_name == WISCAD_NAME:
        bandwidth = SPEED_DOWNLOAD[conn_name]
    else:
        bandwidth = SPEED_UPLOAD[conn_name]

    return intermediate_tensor_size_B / bandwidth


def estimate_communication_energy_asymmetric(
    to_layer: str, conn_name: str, src_name: str, get_input_shape_fn, unit_size=4
) -> float:
    latency = estimate_communication_latency_asymmetric(
        to_layer, conn_name, src_name, get_input_shape_fn, unit_size
    )

    bandwidth = (
        SPEED_DOWNLOAD[conn_name]
        if src_name == WISCAD_NAME
        else SPEED_UPLOAD[conn_name]
    )
    alpha = (
        ALPHA_DOWNLOAD[conn_name]
        if src_name == WISCAD_NAME
        else ALPHA_UPLOAD[conn_name]
    )
    beta = BETA[conn_name]

    alpha = alpha / (1000 * 8)  # uW / MBps
    beta = beta / 1000  # uW
    upload_power = alpha * bandwidth + beta  # uW

    upload_energy = upload_power * latency  # uW * s = uJ

    return upload_energy
