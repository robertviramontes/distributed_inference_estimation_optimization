import torch
import torch.nn as nn
from torchvision_mod.vision_transformer import vit_b_16
import os
import time
import pandas as pd
import gc
from typing import List, Optional
import subprocess
from numpy import repeat
import argparse
import nvpmodel
from profile_common import send_start, send_stop, send_shutdown, generate_combos
import copy


def get_conv_proj(base_state_dict) -> nn.Module:
    vit = vit_b_16(pretrained=False)
    vit.load_state_dict(base_state_dict)
    conv_proj = copy.deepcopy(vit.conv_proj)
    del vit
    return conv_proj


def get_process_input_operations(base_state_dict) -> nn.Module:
    class ProcessInput(nn.Module):
        def __init__(self, p, n, h, w, hidden_size) -> None:
            super().__init__()
            self.p = p  # patch size
            self.n = n  # batch size
            self.h = h  # img height
            self.w = w  # img width
            self.hidden_size = hidden_size

        def forward(self, x):
            # from the PyTorch ViT implementation
            # https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html#vit_b_16
            n_h = self.h // self.p
            n_w = self.w // self.p
            x = x.reshape(self.n, self.hidden_size, n_h * n_w)
            x = x.permute(0, 2, 1)
            class_token = nn.Parameter(
                torch.zeros(1, 1, self.hidden_size, device=x.device)
            )
            batch_class_token = class_token.expand(self.n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            return x

    return ProcessInput(16, 1, 224, 224, 768)


def get_add_pos_embedding(base_state_dict) -> nn.Module:
    class AddPositionEmbedding(nn.Module):
        def __init__(self, base_state_dict) -> None:
            super().__init__()
            model = vit = vit_b_16(pretrained=False)
            vit.load_state_dict(base_state_dict)
            self.pos_embedding = copy.deepcopy(model.encoder.pos_embedding)

        def forward(self, input):
            return input + self.pos_embedding

    return AddPositionEmbedding(base_state_dict)


def get_encoder_layer_t(base_state_dict, t) -> nn.Module:
    vit = vit_b_16(pretrained=False)
    vit.load_state_dict(base_state_dict)
    encoder_layer = copy.deepcopy(vit.encoder.layers[t])

    if t == 11:
        layer_norm = copy.deepcopy(vit.encoder.ln)
        encoder_layer = nn.Sequential(encoder_layer, layer_norm)

    del vit
    return encoder_layer


for i in range(0, 12):
    fn_def = f"""
def get_encoder_layer_{i}(base_state_dict):
    return get_encoder_layer_t(base_state_dict, {i})
    """
    exec(fn_def, globals())


def get_head(base_state_dict) -> nn.Module:
    vit = vit_b_16(pretrained=False)
    vit.load_state_dict(base_state_dict)
    head = copy.deepcopy(vit.heads[0])
    del vit
    return head


def save_base_state_dict(state_dict_filename: str):
    base = vit_b_16(pretrained=True)
    torch.save(base.state_dict(), state_dict_filename)


def get_layers(layers: str, base_state_dict) -> nn.Module:
    layers_arr = layers.split("__")
    components = [(globals()[f"get_{layer}"])(base_state_dict) for layer in layers_arr]

    return nn.Sequential(*components)


def get_input_shape(combo: str):
    first_layer = (combo.split("__"))[0]
    shape_dict = {
        "conv_proj": (1, 3, 224, 224),
        "process_input_operations": (1, 768, 14, 14),
        "add_pos_embedding": (1, 197, 768),
        "encoder_layer_0": (1, 197, 768),
        "encoder_layer_1": (1, 197, 768),
        "encoder_layer_2": (1, 197, 768),
        "encoder_layer_3": (1, 197, 768),
        "encoder_layer_4": (1, 197, 768),
        "encoder_layer_5": (1, 197, 768),
        "encoder_layer_6": (1, 197, 768),
        "encoder_layer_7": (1, 197, 768),
        "encoder_layer_8": (1, 197, 768),
        "encoder_layer_9": (1, 197, 768),
        "encoder_layer_10": (1, 197, 768),
        "encoder_layer_11": (1, 197, 768),
        "head": (1, 197, 768),
    }

    return shape_dict[first_layer]



def get_input_shape_abbv(combo: str):
    '''
    Hack to deal with Gurobi limitations on the length of variable names
    '''
    first_layer = (combo.split("__"))[0]
    shape_dict = {
        "conv_proj": (1, 3, 224, 224),
        "process_input_operations": (1, 768, 14, 14),
        "add_pos_embedding": (1, 197, 768),
        "enc_layer_0": (1, 197, 768),
        "enc_layer_1": (1, 197, 768),
        "enc_layer_2": (1, 197, 768),
        "enc_layer_3": (1, 197, 768),
        "enc_layer_4": (1, 197, 768),
        "enc_layer_5": (1, 197, 768),
        "enc_layer_6": (1, 197, 768),
        "enc_layer_7": (1, 197, 768),
        "enc_layer_8": (1, 197, 768),
        "enc_layer_9": (1, 197, 768),
        "enc_layer_10": (1, 197, 768),
        "enc_layer_11": (1, 197, 768),
        "head": (1, 197, 768),
    }

    return shape_dict[first_layer]


def cpu_profiling(m: nn.Module, x: torch.Tensor, runs: int):
    times = []
    with torch.no_grad():
        for i in range(runs):
            start = time.time()
            _ = m(x)
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
            _ = m(x)
            end.record()
            torch.cuda.synchronize()

            # elapsed_time reports in ms, convert to s for consistency
            times.append(start.elapsed_time(end) / 1000)
    return times


def profiling(combos: List[str], state_dict_filename: str):
    iterations = 1
    runs = 2
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

            x = torch.randn(get_input_shape(combo))
            starts[combo] += [time.time()]
            if torch.cuda.is_available():
                durs[combo] += cuda_profiling(m, x, runs)
            else:
                durs[combo] += cpu_profiling(m, x, runs)
            ends[combo] += [time.time()]
            del m, x
            gc.collect()

    data = pd.DataFrame(
        {
            "layers": durs.keys(),
            "durs": durs.values(),
        }
    )
    data = data.explode("durs")
    data["iter"] = len(combos) * list(repeat(list(range(1, iterations + 1)), runs))
    data["run"] = len(combos) * iterations * list(range(1, runs + 1))

    start_times = pd.DataFrame(
        {"layers": starts.keys(), "iter_start": starts.values()}
    ).explode("iter_start")
    start_times["iter"] = len(combos) * list(range(1, iterations + 1))
    end_times = pd.DataFrame(
        {"layers": ends.keys(), "iter_end": ends.values()}
    ).explode("iter_end")
    end_times["iter"] = len(combos) * list(range(1, iterations + 1))
    times = start_times.merge(end_times, how="left", on=["layers", "iter"])

    data = data.merge(times, how="left", on=["layers", "iter"])

    with open("vit_segments_latencies.csv", "w") as f:
        data.to_csv(f)
        f.flush()
        os.fsync(f.fileno())


def potato_profiling(
    combos: List[str],
    state_dict_filename: str,
    log_server_addr: Optional[str],
    log_server_port: Optional[int],
):
    iterations = 5
    runs = 20
    durs = {}
    starts = {}
    ends = {}

    # set different device settings
    frequencies = [100000, 250000, 500000, 667000, 1000000, 1200000, 1512000]
    for config in frequencies:

        # NOTE it is bad practice, generally, to use shell=True
        # but this is a quick script
        # set the governor
        subprocess.Popen(
            "echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",  # noqa: E501
            shell=True,
        )

        # set the frequency
        subprocess.Popen(
            f"echo {config} | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq",  # noqa: E501
            shell=True,
        )

        # start power logging, noting the frequency config
        if log_server_addr and log_server_port:
            send_start(
                log_server_addr, log_server_port, f"potato_vit_power_{config}.csv"
            )

        for combo in combos:
            durs[combo] = []
            starts[combo] = []
            ends[combo] = []

        for _ in range(iterations):
            # random.shuffle(combos)
            # run latency profiles
            for combo in combos:
                base_state_dict = torch.load(state_dict_filename)
                m = get_layers(combo, base_state_dict)
                m = m.eval()
                del base_state_dict

                x = torch.randn(get_input_shape(combo))
                starts[combo] += [time.time()]
                if torch.cuda.is_available():
                    durs[combo] += cuda_profiling(m, x, runs)
                else:
                    durs[combo] += cpu_profiling(m, x, runs)
                ends[combo] += [time.time()]
                del m, x
                gc.collect()

        data = pd.DataFrame(
            {
                "layers": durs.keys(),
                "durs": durs.values(),
            }
        )
        data = data.explode("durs")
        data["iter"] = len(combos) * list(repeat(list(range(1, iterations + 1)), runs))
        data["run"] = len(combos) * iterations * list(range(1, runs + 1))

        start_times = pd.DataFrame(
            {"layers": starts.keys(), "iter_start": starts.values()}
        ).explode("iter_start")
        start_times["iter"] = len(combos) * list(range(1, iterations + 1))
        end_times = pd.DataFrame(
            {"layers": ends.keys(), "iter_end": ends.values()}
        ).explode("iter_end")
        end_times["iter"] = len(combos) * list(range(1, iterations + 1))
        times = start_times.merge(end_times, how="left", on=["layers", "iter"])

        data = data.merge(times, how="left", on=["layers", "iter"])
        data.to_csv(f"potato_vit_latencies_{config}.csv")

        # end power logging before adjusting device settings
        if log_server_addr and log_server_port:
            send_stop(log_server_addr, log_server_port)

    # reset the governor
    subprocess.Popen(
        "echo schedutil | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",  # noqa: E501
        shell=True,
    )

    # reset the frequency
    subprocess.Popen(
        f"echo {1512000} | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq",  # noqa: E501
        shell=True,
    )


def nano_profiling(
    combos: List[str],
    state_dict_filename: str,
    log_server_addr: Optional[str],
    log_server_port: Optional[int],
):
    iterations = 3
    runs = 20
    device = "cuda"

    for freq in [
        # 76800000,
        # 153600000,
        # 230400000,
        # 307200000,
        # 384000000,
        # 460800000,
        # 537600000,
        # 614400000,
        # 691200000,
        # 768000000,
        # 844800000,
        921600000,
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

        # run the jetson_clocks script to set the GPU to run at configured speed
        # subprocess.run(["jetson_clocks"], check=True)

        # start power logging, noting the frequency config
        if log_server_addr and log_server_port:
            send_start(log_server_addr, log_server_port, f"nano_vit_power_{freq}.csv")

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
                base_state_dict = torch.load(state_dict_filename)
                m = get_layers(combo, base_state_dict)
                m = m.eval()

                del base_state_dict
                m.to(device)
                print(combo)

                # find the correct folder for the samples based on the prefix
                first_layer = combo.split("__")[0]

                samples_base_path = f"vit_samples/{first_layer}"
                inputs = [
                    torch.load(f"{samples_base_path}/{file_name}")
                    for file_name in os.listdir(samples_base_path)
                ]
                inputs = [x.to(device) for x in inputs]

                starts[combo] += [time.time()]

                start_events = [
                    torch.cuda.Event(enable_timing=True) for _ in range(runs)
                ]
                end_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
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
        # # data.to_csv(f"nano_vit_profiles/vit_segments_latencies_{freq}.csv")

        with open(f"nano_vit_profiles/vit_segments_latencies_{freq}.csv", "w") as f:
            data.to_csv(f)
            f.flush()
            os.fsync(f.fileno())

        if log_server_addr and log_server_port:
            send_shutdown(log_server_addr, log_server_port)


def server_profiling(
    combos: List[str],
    state_dict_filename: str,
):
    iterations = 3
    runs = 20
    device = "cuda"

    durs = {}
    starts = {}
    ends = {}
    for combo in combos:
        durs[combo] = []
        starts[combo] = []
        ends[combo] = []

    for iter_num in range(iterations):
        # random.shuffle(combos)
        # run latency profiles
        print(iter_num)
        for combo in combos:
            base_state_dict = torch.load(state_dict_filename)
            m = get_layers(combo, base_state_dict)
            m = m.eval()

            del base_state_dict
            m.to(device)
            print(combo)
            # find the correct folder for the samples based on the prefix
            first_layer = combo.split("__")[0]

            samples_base_path = f"vit_samples/{first_layer}"
            inputs = [
                torch.load(f"{samples_base_path}/{file_name}")
                for file_name in os.listdir(samples_base_path)
            ]
            inputs = [x.to(device) for x in inputs]

            starts[combo] += [time.time()]

            start_events = [torch.cuda.Event(enable_timing=True) for i in range(runs)]
            end_events = [torch.cuda.Event(enable_timing=True) for i in range(runs)]
            with torch.no_grad():
                for i, input in enumerate(inputs):
                    start_events[i].record()
                    _ = m(input)
                    end_events[i].record()

            # wait for all queued events to finish
            torch.cuda.synchronize()
            # elapsed_time reports in ms, convert to s for consistency
            times = [
                start_events[i].elapsed_time(end_events[i]) / 1000 for i in range(runs)
            ]
            durs[combo] += times

            ends[combo] += [time.time()]
            del m, inputs
            gc.collect()

    data = pd.DataFrame(
        {
            "layers": durs.keys(),
            "durs": durs.values(),
        }
    )
    data = data.explode("durs")
    data["iter"] = len(combos) * list(repeat(list(range(1, iterations + 1)), runs))
    data["run"] = len(combos) * iterations * list(range(1, runs + 1))

    start_times = pd.DataFrame(
        {"layers": starts.keys(), "iter_start": starts.values()}
    ).explode("iter_start")
    start_times["iter"] = len(combos) * list(range(1, iterations + 1))
    end_times = pd.DataFrame(
        {"layers": ends.keys(), "iter_end": ends.values()}
    ).explode("iter_end")
    end_times["iter"] = len(combos) * list(range(1, iterations + 1))
    times = start_times.merge(end_times, how="left", on=["layers", "iter"])

    data = data.merge(times, how="left", on=["layers", "iter"])

    with open("wiscad_vit_segments_latencies.csv", "w") as f:
        data.to_csv(f)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    state_dict_filename = "vit_state_dict.pt"

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
        "conv_proj",
        "process_input_operations",
        "add_pos_embedding",
        "encoder_layer_0",
        "encoder_layer_1",
        "encoder_layer_2",
        "encoder_layer_3",
        "encoder_layer_4",
        "encoder_layer_5",
        "encoder_layer_6",
        "encoder_layer_7",
        "encoder_layer_8",
        "encoder_layer_9",
        "encoder_layer_10",
        "encoder_layer_11",
        "head",
    ]

    combos = generate_combos(combo_bases, len(combo_bases), "__")

    if args.device is not None:
        if args.device.lower() == "nano":
            nano_profiling(
                combos, state_dict_filename, args.log_server_addr, args.log_server_port
            )
        elif "potato" in args.device.lower():
            potato_profiling(
                combos, state_dict_filename, args.log_server_addr, args.log_server_port
            )
        elif "cloud" in args.device.lower() or "server" in args.device.lower():
            server_profiling(combos, state_dict_filename)
    else:
        profiling(combos, state_dict_filename)
