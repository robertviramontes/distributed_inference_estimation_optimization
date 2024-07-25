import torch
import torch.nn as nn
from torchvision.models import vgg11
import os
import time
import pandas as pd
import gc
from typing import List, Optional
import subprocess
from numpy import repeat
import argparse
import nvpmodel
from profile_common import (
    send_start,
    send_stop,
    send_shutdown,
    generate_combos,
    potato_profiling,
)


def get_conv1(base_state_dict) -> nn.Module:
    conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    )
    conv1.load_state_dict(
        {
            "0.weight": base_state_dict["features.0.weight"],
            "0.bias": base_state_dict["features.0.bias"],
        }
    )
    return conv1


def get_conv2(base_state_dict) -> nn.Module:
    conv2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    )
    conv2.load_state_dict(
        {
            "0.weight": base_state_dict["features.3.weight"],
            "0.bias": base_state_dict["features.3.bias"],
        }
    )
    return conv2


def get_conv3(base_state_dict) -> nn.Module:
    conv3 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
    )
    conv3.load_state_dict(
        {
            "0.weight": base_state_dict["features.6.weight"],
            "0.bias": base_state_dict["features.6.bias"],
        }
    )
    return conv3


def get_conv4(base_state_dict) -> nn.Module:
    conv4 = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    )
    conv4.load_state_dict(
        {
            "0.weight": base_state_dict["features.8.weight"],
            "0.bias": base_state_dict["features.8.bias"],
        }
    )
    return conv4


def get_conv5(base_state_dict) -> nn.Module:
    conv5 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
    )
    conv5.load_state_dict(
        {
            "0.weight": base_state_dict["features.11.weight"],
            "0.bias": base_state_dict["features.11.bias"],
        }
    )
    return conv5


def get_conv6(base_state_dict) -> nn.Module:
    conv6 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    )
    conv6.load_state_dict(
        {
            "0.weight": base_state_dict["features.13.weight"],
            "0.bias": base_state_dict["features.13.bias"],
        }
    )
    return conv6


def get_conv7(base_state_dict) -> nn.Module:
    conv7 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
    )
    conv7.load_state_dict(
        {
            "0.weight": base_state_dict["features.16.weight"],
            "0.bias": base_state_dict["features.16.bias"],
        }
    )
    return conv7


def get_conv8(base_state_dict) -> nn.Module:
    conv8 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        nn.Flatten(),
    )
    conv8.load_state_dict(
        {
            "0.weight": base_state_dict["features.18.weight"],
            "0.bias": base_state_dict["features.18.bias"],
        }
    )
    return conv8


def get_fc1(base_state_dict) -> nn.Module:
    fc1 = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
    )
    fc1.load_state_dict(
        {
            "0.weight": base_state_dict["classifier.0.weight"],
            "0.bias": base_state_dict["classifier.0.bias"],
        }
    )
    return fc1


def get_fc2(base_state_dict) -> nn.Module:
    fc2 = nn.Sequential(
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
    )
    fc2.load_state_dict(
        {
            "0.weight": base_state_dict["classifier.3.weight"],
            "0.bias": base_state_dict["classifier.3.bias"],
        }
    )
    return fc2


def get_fc3(base_state_dict) -> nn.Module:
    fc3 = nn.Sequential(nn.Linear(in_features=4096, out_features=1000, bias=True))
    fc3.load_state_dict(
        {
            "0.weight": base_state_dict["classifier.6.weight"],
            "0.bias": base_state_dict["classifier.6.bias"],
        }
    )
    return fc3


def save_base_state_dict(state_dict_filename: str):
    base = vgg11(pretrained=True)
    torch.save(base.state_dict(), state_dict_filename)


def get_layers(layers: str, base_state_dict) -> nn.Module:
    layers_arr = layers.split("_")
    components = [(globals()[f"get_{layer}"])(base_state_dict) for layer in layers_arr]

    return nn.Sequential(*components)


def get_input_shape(combo: str):
    first_layer = (combo.split("_"))[0]
    shape_dict = {
        "conv1": (1, 3, 224, 224),
        "conv2": (1, 64, 112, 112),
        "conv3": (1, 128, 56, 56),
        "conv4": (1, 256, 56, 56),
        "conv5": (1, 256, 28, 28),
        "conv6": (1, 512, 28, 28),
        "conv7": (1, 512, 14, 14),
        "conv8": (1, 512, 14, 14),
        "fc1": (1, 25088),
        "fc2": (1, 4096),
        "fc3": (1, 4096),
    }

    return shape_dict[first_layer]


def cpu_profiling(m: nn.Module, x: torch.Tensor, runs: int):
    times = []
    with torch.no_grad():
        for i in range(runs):
            start = time.time()
            y = m(x)
            end = time.time()
            times.append(end - start)
    return times


def cuda_profiling(m: nn.Module, x: torch.Tensor, runs: int):
    device = "cuda"
    x = x.to(device)
    m = m.to(device)

    times = []
    with torch.no_grad():
        for i in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            y = m(x)
            end.record()
            torch.cuda.synchronize()

            # elapsed_time reports in ms, convert to s for consistency
            times.append(start.elapsed_time(end) / 1000)
    return times


def profiling(combos: List[str], state_dict_filename: str):
    iterations = 1
    runs = 20
    durs = {}
    for combo in combos:
        durs[combo] = []

    for _ in range(iterations):
        # random.shuffle(combos)
        # run latency profiles
        for combo in combos:
            base_state_dict = torch.load(state_dict_filename)
            m = get_layers(combo, base_state_dict)
            m = m.eval()
            del base_state_dict

            x = torch.randn(get_input_shape(combo))
            if torch.cuda.is_available():
                durs[combo] += cuda_profiling(m, x, runs)
            else:
                durs[combo] += cpu_profiling(m, x, runs)

            del m, x
            gc.collect()

    data = pd.DataFrame({"layers": durs.keys(), "durs": durs.values()})
    data = data.explode("durs")
    data.to_csv("vgg11_segments_latencies.csv")


def nano_profiling(
    combos: List[str],
    state_dict_filename: str,
    log_server_addr: Optional[str],
    log_server_port: Optional[int],
):
    iterations = 5
    runs = 20
    device = "cuda"

    for freq in [
        921600000,
        844800000,
        768000000,
        691200000,
        614400000,
        537600000,
        460800000,
        # 384000000,
        # 307200000,
        # 230400000,
        # 153600000,
        # 76800000,
    ]:
        # create the nvpmodel file and set the gpu clocks
        nvp = nvpmodel.PARAM_DEFINITIONS + nvpmodel.generate_power_model_definition(
            freq
        )
        with open(f"{freq}_nvp.conf", "w") as file:
            file.write(nvp)

        nvp_args = ["nvpmodel", "-m", "0", "-f", "default_nvp.conf"]
        subprocess.run(nvp_args, check=True)

        nvp_args = ["nvpmodel", "-m", "0", "-f", f"{freq}_nvp.conf"]
        subprocess.run(nvp_args, check=True)

        # start power logging, noting the frequency config
        if log_server_addr and log_server_port:
            send_start(log_server_addr, log_server_port, f"nano_vgg11_power_fixed_{freq}.csv")

        time.sleep(5)  # let logging server start up

        durs = {}
        starts = {}
        ends = {}
        for combo in combos:
            durs[combo] = []
            starts[combo] = []
            ends[combo] = []

        for _ in range(iterations):
            # random.shuffle(combos)
            # run latency profiles
            for combo in combos:
                print(combo)
                base_state_dict = torch.load(state_dict_filename)
                m = get_layers(combo, base_state_dict)
                m = m.eval()

                del base_state_dict
                m.to(device)

                # find the correct folder for the samples based on the prefix
                first_layer = combo.split("_")[0]

                samples_base_path = f"vgg11_samples/{first_layer}"
                inputs = [
                    torch.load(f"{samples_base_path}/{file_name}")
                    for file_name in os.listdir(samples_base_path)
                ]
                inputs = [x.unsqueeze(0) if len(x.size()) == 3 else x for x in inputs]
                inputs = [x.to(device) for x in inputs]

                start_events = [
                    torch.cuda.Event(enable_timing=True) for i in range(runs)
                ]
                end_events = [torch.cuda.Event(enable_timing=True) for i in range(runs)]
                with torch.no_grad():
                    for i, input in enumerate(inputs):
                        starts[combo] += [time.time()]
                        start_events[i].record()
                        _ = m(input)
                        end_events[i].record()
                        torch.cuda.synchronize()
                        ends[combo] += [time.time()]

                # wait for all queued events to finish
                torch.cuda.synchronize()
                # elapsed_time reports in ms, convert to s for consistency
                times = [
                    start_events[i].elapsed_time(end_events[i]) / 1000
                    for i in range(runs)
                ]
                durs[combo] += times

                ends[combo] += [time.time()]
                del m, inputs
                gc.collect()

        # end power logging before adjusting device settings
        if log_server_addr and log_server_port:
            send_stop(log_server_addr, log_server_port)

        data = pd.DataFrame(
            {
                "layers": durs.keys(),
                "durs": durs.values(),
                "starts": starts.values(),
                "ends": ends.values()
            }
        )
        # data = data.explode("durs")
        # data["iter"] = len(combos) * list(repeat(list(range(1, iterations + 1)), runs))
        # data["run"] = len(combos) * iterations * list(range(1, runs + 1))

        # start_times = pd.DataFrame(
        #     {"layers": starts.keys(), "iter_start": starts.values()}
        # ).explode("iter_start")
        # start_times["iter"] = len(combos) * list(range(1, iterations + 1))
        # end_times = pd.DataFrame(
        #     {"layers": ends.keys(), "iter_end": ends.values()}
        # ).explode("iter_end")
        # end_times["iter"] = len(combos) * list(range(1, iterations + 1))
        # times = start_times.merge(end_times, how="left", on=["layers", "iter"])

        # data = data.merge(times, how="left", on=["layers", "iter"])

        with open(f"nano_vgg11_profiles_fixed/vgg11_segments_latencies_{freq}.csv", "w") as f:
            data.to_csv(f)
            f.flush
            os.fsync(f.fileno())

        if log_server_addr and log_server_port:
            send_shutdown(log_server_addr, log_server_port)


if __name__ == "__main__":
    state_dict_filename = "vgg11_state_dict.pt"

    if not os.path.isfile(state_dict_filename):
        save_base_state_dict(state_dict_filename)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device")
    parser.add_argument("-i", "--iterations", default=5, type=int)
    parser.add_argument("-r", "--runs", default=20, type=int)
    parser.add_argument("--log-server-addr", default=None)
    parser.add_argument("--log-server-port", default=None, type=int)
    args = parser.parse_args()

    combo_bases = [
        "conv1",
        "conv2",
        "conv3",
        "conv4",
        "conv5",
        "conv6",
        "conv7",
        "conv8",
        "fc1",
        "fc2",
        "fc3",
    ]

    combos = generate_combos(combo_bases, len(combo_bases))

    if args.device is not None:
        if args.device.lower() == "nano":
            nano_profiling(
                combos, state_dict_filename, args.log_server_addr, args.log_server_port
            )
        elif "potato" in args.device.lower():
            potato_profiling(
                combos,
                state_dict_filename,
                get_layers,
                "vgg11",
                "vgg11_samples",
                args.log_server_addr,
                args.log_server_port,
            )
    else:
        profiling(combos, state_dict_filename, None, None)
