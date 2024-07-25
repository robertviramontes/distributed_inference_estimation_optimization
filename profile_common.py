import socket
from typing import List, Optional
import subprocess
import torch
import os
import time
import pandas as pd
import gc
from numpy import repeat


def send_start(server_address, server_port, filename):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((server_address, server_port))
        s.sendall((f"start,{filename}").encode())
        _ = s.recv(1024)


def send_stop(server_address, server_port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((server_address, server_port))
        s.sendall(b"stop")
        _ = s.recv(1024)


def send_shutdown(server_address, server_port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((server_address, server_port))
        s.sendall(b"shutdown")
        _ = s.recv(1024)


def generate_combos(bases: List[str], max_block_size=3, join_char="_") -> List[str]:
    combos = bases.copy()  # make sure copy, not ref, since we modify combos in loop
    for block_size in range(2, max_block_size + 1):
        num_blocks_of_size = len(bases) - (block_size - 1)
        for i in range(num_blocks_of_size):
            bases_in_combo = bases[i : i + block_size]
            combos.append(join_char.join(bases_in_combo))
    return combos


def potato_profiling(
    combos: List[str],
    state_dict_filename: str,
    get_layers_fn,
    network_name: str,
    samples_base_path,
    log_server_addr: Optional[str],
    log_server_port: Optional[int],
):
    iterations = 5
    runs = 20
    durs = {}
    starts = {}
    ends = {}

    # set different device settings
    # frequencies = [1200000, 1000000, 667000, 500000, 250000, 100000]
    frequencies = [1200000, 1000000, 667000]
    for config in frequencies:

        # NOTE it is bad practice, generally, to use shell=True, but this is a quick script
        # set the governor
        subprocess.Popen(
            "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
            shell=True,
        )

        # set the frequency
        set_freq_proc = subprocess.Popen(
            f"echo {config} | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq",
            shell=True,
        )
        set_freq_proc.wait()

        # start power logging, noting the frequency config
        if log_server_addr and log_server_port:
            send_start(
                log_server_addr,
                log_server_port,
                f"potato_{network_name}_power_fixed_{config}.csv",
            )

        for combo in combos:
            durs[combo] = []
            starts[combo] = []
            ends[combo] = []

        for iter in range(iterations):
            print(iter)
            # random.shuffle(combos)
            # run latency profiles
            for combo in combos:
                first_layer = combo.split("_")[0]

                base_state_dict = torch.load(state_dict_filename)
                m = get_layers_fn(combo, base_state_dict)
                m = m.eval()
                del base_state_dict

                samples_path = f"{samples_base_path}/{first_layer}"
                inputs = [
                    torch.load(f"{samples_path}/{file_name}")
                    for file_name in os.listdir(samples_path)
                ]
                inputs = [x.unsqueeze(0) if len(x.size()) == 3 else x for x in inputs]

                with torch.no_grad():
                    for i, input in enumerate(inputs):
                        if i >= runs:
                            break
                        # input = torch.load(input_path)
                        # if len(input.size()) == 3: input = input.unsqueeze(0)
                        # print(f"iter: {iter}  run: {i}")
                        starts[combo] += [time.time_ns()]
                        _ = m(input)
                        ends[combo] += [time.time_ns()]
                time.sleep(10)        

                durs[combo] = [
                    ends[combo][i] - starts[combo][i] for i in range(len(starts[combo]))
                ]
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
                "ends": ends.values(),
            }
        )
        data.to_pickle("temp.pickle")
        data = data.explode(["durs", "starts", "ends"])
        data["iter"] = len(combos) * list(repeat(list(range(1, iterations + 1)), runs))
        data["run"] = len(combos) * iterations * list(range(1, runs + 1))

        with open(f"potato_{network_name}_latencies_fixed_{config}.csv", "w") as f:
            data.to_csv(f)
            f.flush
            os.fsync(f.fileno())

        if log_server_addr and log_server_port:
            send_shutdown(log_server_addr, log_server_port)

    # reset the governor
    subprocess.Popen(
        "echo schedutil | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
        shell=True,
    )

    # reset the frequency
    subprocess.Popen(
        f"echo {1512000} | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq",
        shell=True,
    )
