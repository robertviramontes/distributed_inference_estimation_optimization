import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from os import path

from ilp_solver_common import (
    trimmean,
    potato_trimmean,
    estimate_communication_latency,
    estimate_communication_latency_asymmetric,
    estimate_communication_energy_asymmetric,
    LEPOTATO_NAME,
    NANO_NAME,
    WISCAD_NAME,
)


def get_input_shape(combo: str):
    first_layer = (combo.split("_"))[0]
    shape_dict = {
        "conv1": (1, 3, 224, 224),
        "conv2": (1, 64, 27, 27),
        "conv3": (1, 192, 13, 13),
        "conv4": (1, 384, 13, 13),
        "conv5": (1, 256, 13, 13),
        "fc1": (1, 9216),
        "fc2": (1, 4096),
        "fc3": (1, 4096),
    }

    return shape_dict[first_layer]


def estimate_communication_energy(
    to_layer: str, bandwidth: float, unit_size=4
) -> float:
    if to_layer == "output":
        intermediate_tensor_size = [1]
    else:
        intermediate_tensor_size = get_input_shape(to_layer)

    intermediate_tensor_size_B = np.prod(intermediate_tensor_size) * unit_size
    latency = intermediate_tensor_size_B / bandwidth

    # see JointDNN for evidence
    if bandwidth <= (2.0275 * 1e6) / 8.0:  # 3G
        alpha_upload = 868.98  # mW / Mbps
        beta = 817.88  # mW
    elif bandwidth <= (13.76 * 1e6) / 8.0:  # 4G
        alpha_upload = 438.39
        beta = 1288.04
    else:  # WIFI
        alpha_upload = 283.17
        beta = 132.86

    alpha_upload = alpha_upload / (1000 * 8)  # uW / MBps
    beta = beta / 1000  # uW
    upload_power = alpha_upload * bandwidth + beta  # uW

    upload_energy = upload_power * latency  # uW * s = uJ

    return upload_energy


def run_optimization(bandwidth: float, max_layers_in_block: int):
    # load in latency values
    lepotato = pd.read_csv("lepotato_alexnet_segments_latencies.csv", index_col=[0])
    lepotato["num_blocks"] = lepotato["layers"].apply(lambda x: len(x.split("_")))
    lepotato_vals = lepotato[lepotato["num_blocks"] <= max_layers_in_block]
    lepotato_vals = (
        lepotato_vals.drop(columns=["num_blocks"])
        .groupby(["layers"])
        .apply(trimmean, 0.2)
    )

    wiscad = pd.read_csv("wiscad_alexnet_segments_latencies.csv", index_col=[0])
    wiscad["durs"] = wiscad["durs"] / 1000
    wiscad["num_blocks"] = wiscad["layers"].apply(lambda x: len(x.split("_")))
    wiscad_vals = wiscad[wiscad["num_blocks"] <= max_layers_in_block]
    wiscad_vals = (
        wiscad_vals.drop(columns=["num_blocks"])
        .groupby(["layers"])
        .apply(trimmean, 0.2)
    )

    computation_dict = {}
    assignment_list = []
    layers = []

    for layer_name, latency in lepotato_vals.items():
        computation_dict[(layer_name, LEPOTATO_NAME)] = latency
        if len(layer_name.split("_")) == 1:
            assignment_list.append((layer_name, LEPOTATO_NAME))
            layers.append(layer_name)

    for layer_name, latency in wiscad_vals.items():
        computation_dict[(layer_name, WISCAD_NAME)] = latency
        if len(layer_name.split("_")) == 1:
            assignment_list.append((layer_name, WISCAD_NAME))

    blocks_to_consider = list(wiscad_vals.index)

    opt_model = gp.Model("AlexNet")

    # computation times
    computations, comp_latencies = gp.multidict(computation_dict)

    layer_device = gp.tuplelist(
        [("input", LEPOTATO_NAME)] + assignment_list + [("output", LEPOTATO_NAME)]
    )

    # generate communication combinations
    comm_combos = []
    for i, layer in enumerate(layer_device):
        if i == len(layer_device) - 1:
            break
        if layer[0] == "input":
            next_layer = layers[0]
        elif layer[0] == layers[-1]:
            next_layer = "output"
        else:
            next_layer = layers[layers.index(layer[0]) + 1]

        devices_that_next_layer_can_sent_to = [
            device
            for lyr, device in layer_device
            if lyr == next_layer and device != layer[1]
        ]
        comm_combos += [
            (layer[0], next_layer, layer[1], device)
            for device in devices_that_next_layer_can_sent_to
        ]

    comm_latencies = {}
    # communication times
    for comm_combo in comm_combos:
        comm_latencies[comm_combo] = estimate_communication_latency(
            comm_combo[1], bandwidth, get_input_shape
        )

    communications, comm_latency = gp.multidict(comm_latencies)

    # add assignment variable
    y = opt_model.addVars(computations, name="blocks", vtype=gp.GRB.BINARY)
    z = opt_model.addVars(layer_device, name="assignment", vtype=gp.GRB.BINARY)
    c = opt_model.addVars(communications, name="communication", vtype=gp.GRB.BINARY)

    # layer can only be assigned to one device
    opt_model.addConstrs((z.sum(layer, "*") == 1 for layer in layers))

    # input on lepotato
    opt_model.addConstr(z["input", LEPOTATO_NAME] == 1)
    # output on lepotato
    opt_model.addConstr(z["output", LEPOTATO_NAME] == 1)

    # find the layer assignment, using the block that the layer is executed in
    for i, layer in enumerate(layers):
        blocks_containing_layer = [
            block for block in blocks_to_consider if layer in block
        ]
        opt_model.addConstr(
            z[(layer, LEPOTATO_NAME)]
            == gp.quicksum(
                [y[block, LEPOTATO_NAME] for block in blocks_containing_layer]
            )
        )
        opt_model.addConstr(
            z[(layer, WISCAD_NAME)]
            == gp.quicksum([y[block, WISCAD_NAME] for block in blocks_containing_layer])
        )

    # constr to solve the communication var
    for combo in comm_combos:
        opt_model.addConstr(
            c[combo] == gp.and_([z[combo[0], combo[2]], z[combo[1], combo[3]]])
        )

    edge_constr = opt_model.addVar(vtype=gp.GRB.INTEGER, name="edge_blk_cnt")
    cloud_constr = opt_model.addVar(vtype=gp.GRB.INTEGER, name="cld_blk_cnt")

    opt_model.addConstr(
        edge_constr
        >= (
            gp.quicksum(
                [
                    z[combo]
                    for combo in z.keys()
                    if LEPOTATO_NAME in combo
                    and "input" not in combo
                    and "output" not in combo
                ]
            )
            / max_layers_in_block
        ),
        name="edge_constr_lower_bound",
    )
    opt_model.addConstr(
        edge_constr + 1e-3
        <= (
            gp.quicksum(
                [
                    z[combo]
                    for combo in z.keys()
                    if LEPOTATO_NAME in combo
                    and "input" not in combo
                    and "output" not in combo
                ]
            )
            / max_layers_in_block
        )
        + 1,
        name="edge_constr_upper_bound",
    )

    opt_model.addConstr(
        cloud_constr
        >= (
            gp.quicksum(
                [
                    z[combo]
                    for combo in z.keys()
                    if WISCAD_NAME in combo
                    and "input" not in combo
                    and "output" not in combo
                ]
            )
            / max_layers_in_block
        ),
        name="cloud_constr_lower_bound",
    )
    opt_model.addConstr(
        cloud_constr + 1e-3
        <= (
            gp.quicksum(
                [
                    z[combo]
                    for combo in z.keys()
                    if WISCAD_NAME in combo
                    and "input" not in combo
                    and "output" not in combo
                ]
            )
            / max_layers_in_block
        )
        + 1,
        name="cloud_constr_upper_bound",
    )

    opt_model.addConstr(
        edge_constr
        # max_layers_per_device
        >= gp.quicksum(
            [
                y[combo]
                for combo in y.keys()
                if LEPOTATO_NAME in combo
                and "input" not in combo
                and "output" not in combo
            ]
        )
    )
    opt_model.addConstr(
        1
        + (
            gp.quicksum(
                [
                    z[combo]
                    for combo in z.keys()
                    if WISCAD_NAME in combo
                    and "input" not in combo
                    and "output" not in combo
                ]
            )
            / max_layers_in_block
        )
        # max_layers_per_device
        >= gp.quicksum(
            [
                y[combo]
                for combo in y.keys()
                if WISCAD_NAME in combo
                and "input" not in combo
                and "output" not in combo
            ]
        )
    )

    opt_model.setObjective(y.prod(comp_latencies) + c.prod(comm_latencies))
    opt_model.write("out.lp")

    opt_model.optimize()

    for soln in range(opt_model.getAttr(gp.GRB.Attr.SolCount)):
        print(f"solution {soln}")
        for var in y:
            opt_model.setParam(gp.GRB.Param.SolutionNumber, soln)
            if y[var].xn > 0:
                print(f"\t{var}: {y[var].xn}")

    with open("alexnet_experiments.txt", "a") as log:
        log.write(f"max block size: {max_layers_in_block}\n")
        log.write(f"bandwidth: {bandwidth}\n")
        for soln in range(opt_model.getAttr(gp.GRB.Attr.SolCount)):
            opt_model.setParam(gp.GRB.Param.SolutionNumber, soln)

            log.write(f"\tsolution {soln}\n")
            log.write(f"\tsolution latency: {opt_model.poolObjVal}\n")

            for var in y:
                if y[var].xn > 0:
                    log.write(f"\t\t{var}: {y[var].xn}\n")

    if max_layers_in_block == 8:
        # do jalad-like optimization now that we have access to all
        print(
            f"jalad all edge: {comp_latencies['conv1_conv2_conv3_conv4_conv5_fc1_fc2_fc3', LEPOTATO_NAME]}"
        )

        # 7 edge, 1 cloud
        l = (
            comp_latencies["conv1_conv2_conv3_conv4_conv5_fc1_fc2", LEPOTATO_NAME]
            + comp_latencies["fc3", WISCAD_NAME]
            + comm_latencies["fc2", "fc3", LEPOTATO_NAME, WISCAD_NAME]
            + comm_latencies["fc3", "output", WISCAD_NAME, LEPOTATO_NAME]
        )
        print(f"jalad 7 edge 1 cloud: {l}")
        l = (
            comp_latencies["conv1_conv2_conv3_conv4_conv5_fc1", LEPOTATO_NAME]
            + comp_latencies["fc2_fc3", WISCAD_NAME]
            + comm_latencies["fc1", "fc2", LEPOTATO_NAME, WISCAD_NAME]
            + comm_latencies["fc3", "output", WISCAD_NAME, LEPOTATO_NAME]
        )
        print(f"jalad 6 edge 2 cloud: {l}")

        l = (
            comp_latencies["conv1_conv2_conv3_conv4_conv5", LEPOTATO_NAME]
            + comp_latencies["fc1_fc2_fc3", WISCAD_NAME]
            + comm_latencies["conv5", "fc1", LEPOTATO_NAME, WISCAD_NAME]
            + comm_latencies["fc3", "output", WISCAD_NAME, LEPOTATO_NAME]
        )
        print(f"jalad 5 edge 3 cloud: {l}")

        l = (
            comp_latencies["conv1_conv2_conv3_conv4", LEPOTATO_NAME]
            + comp_latencies["conv5_fc1_fc2_fc3", WISCAD_NAME]
            + comm_latencies["conv4", "conv5", LEPOTATO_NAME, WISCAD_NAME]
            + comm_latencies["fc3", "output", WISCAD_NAME, LEPOTATO_NAME]
        )
        print(f"jalad 4 edge 4 cloud: {l}")

        l = (
            comp_latencies["conv1_conv2_conv3", LEPOTATO_NAME]
            + comp_latencies["conv4_conv5_fc1_fc2_fc3", WISCAD_NAME]
            + comm_latencies["conv3", "conv4", LEPOTATO_NAME, WISCAD_NAME]
            + comm_latencies["fc3", "output", WISCAD_NAME, LEPOTATO_NAME]
        )
        print(f"jalad 3 edge 5 cloud: {l}")

        l = (
            comp_latencies["conv1_conv2", LEPOTATO_NAME]
            + comp_latencies["conv3_conv4_conv5_fc1_fc2_fc3", WISCAD_NAME]
            + comm_latencies["conv2", "conv3", LEPOTATO_NAME, WISCAD_NAME]
            + comm_latencies["fc3", "output", WISCAD_NAME, LEPOTATO_NAME]
        )
        print(f"jalad 2 edge 6 cloud: {l}")

        l = (
            comp_latencies["conv1", LEPOTATO_NAME]
            + comp_latencies["conv2_conv3_conv4_conv5_fc1_fc2_fc3", WISCAD_NAME]
            + comm_latencies["conv1", "conv2", LEPOTATO_NAME, WISCAD_NAME]
            + comm_latencies["fc3", "output", WISCAD_NAME, LEPOTATO_NAME]
        )
        print(f"jalad 1 edge 7 cloud: {l}")

        l = (
            comp_latencies["conv1_conv2_conv3_conv4_conv5_fc1_fc2_fc3", WISCAD_NAME]
            + comm_latencies["input", "conv1", LEPOTATO_NAME, WISCAD_NAME]
            + comm_latencies["fc3", "output", WISCAD_NAME, LEPOTATO_NAME]
        )
        print(f"jalad 0 edge 8 cloud: {l}")

    if max_layers_in_block == 8:
        # generate an pedram model

        pedram_model = opt_model.copy()
        pedram_model.remove(pedram_model.getConstrByName("edge_constr_lower_bound"))
        pedram_model.remove(pedram_model.getConstrByName("edge_constr_upper_bound"))
        pedram_model.remove(pedram_model.getConstrByName("cloud_constr_lower_bound"))
        pedram_model.remove(pedram_model.getConstrByName("cloud_constr_upper_bound"))

        pedram_model.optimize()

        for var in pedram_model.getVars():
            if var.x > 0.1:
                print(f"{var}: {var.x}")
        print(pedram_model.objVal)


def run_energy_optimization(bandwidth: float, max_layers_in_block: int, conn_name=None):
    # load in profiles
    USING_NANO = True
    EDGE_NAME = NANO_NAME if USING_NANO else LEPOTATO_NAME

    lepotato = pd.read_parquet("nano_alexnet_profiles_fixed/latency_and_energy.parquet")
    lepotato["num_blocks"] = lepotato["layers"].apply(lambda x: len(x.split("_")))
    if not USING_NANO:
        lepotato["durs"] = lepotato["durs"] / 1e9
        lepotato = lepotato[lepotato["run"] < 5]
        lepotato = lepotato[lepotato["frequency"] < 1512000]

    # filter to only the bundle sizes of interest
    lepotato_vals = lepotato[lepotato["num_blocks"] <= max_layers_in_block]
    lepotato_vals = lepotato_vals[lepotato_vals["run_energy"] > 1]
    lepotato_vals = lepotato_vals[lepotato_vals["iter"] > 1]

    lepotato_vals = lepotato_vals.drop(columns=["num_blocks"]).groupby(
        ["layers", "frequency"], as_index=False
    )

    if not USING_NANO:
        lepotato_latencies = lepotato_vals.apply(potato_trimmean, 0.2)
    else:
        lepotato_latencies = lepotato_vals["durs"].apply(trimmean, 0.2)
    lepotato_freqs = lepotato["frequency"].unique()

    wiscad = pd.read_csv("wiscad_alexnet_segments_latencies.csv", index_col=[0])
    wiscad["durs"] = wiscad["durs"] / 1000
    wiscad["num_blocks"] = wiscad["layers"].apply(lambda x: len(x.split("_")))
    wiscad_vals = wiscad[wiscad["num_blocks"] <= max_layers_in_block]
    wiscad_vals = (
        wiscad_vals.drop(columns=["num_blocks"])
        .groupby(["layers"])
        .apply(trimmean, 0.2)
    )

    computation_dict = {}
    assignment_list = []
    layers = []

    for _, row in lepotato_latencies.iterrows():
        layer_name = row["layers"]
        frequency = row["frequency"]
        latency = row["durs"]
        computation_dict[(layer_name, frequency, EDGE_NAME)] = latency
        if len(layer_name.split("_")) == 1:
            if not ((layer_name, EDGE_NAME) in assignment_list):
                assignment_list.append((layer_name, EDGE_NAME))
            if not (layer_name in layers):
                layers.append(layer_name)
        del layer_name, frequency, latency

    for layer_name, latency in wiscad_vals.items():
        frequency = -1
        computation_dict[(layer_name, frequency, WISCAD_NAME)] = latency
        if len(layer_name.split("_")) == 1:
            assignment_list.append((layer_name, WISCAD_NAME))

    blocks_to_consider = list(wiscad_vals.index)

    opt_model = gp.Model("AlexNet")

    # computation times
    computations, comp_latencies = gp.multidict(computation_dict)

    layer_device = gp.tuplelist(
        [("input", EDGE_NAME)] + assignment_list + [("output", EDGE_NAME)]
    )

    # generate communication combinations
    comm_combos = []
    for i, layer in enumerate(layer_device):
        if i == len(layer_device) - 1:
            break
        if layer[0] == "input":
            next_layer = layers[0]
        elif layer[0] == layers[-1]:
            next_layer = "output"
        else:
            next_layer = layers[layers.index(layer[0]) + 1]

        devices_that_next_layer_can_sent_to = [
            device
            for lyr, device in layer_device
            if lyr == next_layer and device != layer[1]
        ]
        comm_combos += [
            (layer[0], next_layer, layer[1], device)
            for device in devices_that_next_layer_can_sent_to
        ]

    comm_latencies = {}
    comm_energy_dict = {}

    # communication times
    for comm_combo in comm_combos:
        comm_latencies[comm_combo] = estimate_communication_latency_asymmetric(
            comm_combo[1], conn_name, comm_combo[2], get_input_shape
        )
        comm_energy_dict[comm_combo] = estimate_communication_energy_asymmetric(
            comm_combo[1], conn_name, comm_combo[2], get_input_shape
        )

    communications, comm_latency = gp.multidict(comm_latencies)

    # create a useful dict for looking at the energies
    comp_energy_dict = {}
    lepotato_energies = lepotato_vals["run_energy"].mean()
    for _, row in lepotato_energies.iterrows():
        layer_name = row["layers"]
        frequency = row["frequency"]
        energy = row["run_energy"]
        comp_energy_dict[(layer_name, frequency, EDGE_NAME)] = energy

        del layer_name, frequency, energy

    # add assignment variable
    # y is the assignment variable for bundles
    y = opt_model.addVars(computations, name="blocks", vtype=gp.GRB.BINARY)
    # z is the assignment variable for each layer
    z = opt_model.addVars(layer_device, name="assignment", vtype=gp.GRB.BINARY)
    # c is the assignment variable for energy
    c = opt_model.addVars(communications, name="communication", vtype=gp.GRB.BINARY)

    # layer can only be assigned to one device
    opt_model.addConstrs((z.sum(layer, "*") == 1 for layer in layers))

    # input on lepotato
    opt_model.addConstr(z["input", EDGE_NAME] == 1)
    # output on lepotato
    opt_model.addConstr(z["output", EDGE_NAME] == 1)

    # find the layer assignment, using the block that the layer is executed in
    for i, layer in enumerate(layers):
        blocks_containing_layer = [
            block for block in blocks_to_consider if layer in block
        ]
        opt_model.addConstr(
            z[(layer, EDGE_NAME)]
            == gp.quicksum(
                [
                    y[block, freq, EDGE_NAME]
                    for block in blocks_containing_layer
                    for freq in lepotato_freqs
                ]
            )
        )
        opt_model.addConstr(
            z[(layer, WISCAD_NAME)]
            == gp.quicksum(
                [y[block, -1, WISCAD_NAME] for block in blocks_containing_layer]
            )
        )

    # constr to solve the communication var
    for combo in comm_combos:
        opt_model.addConstr(
            c[combo] == gp.and_([z[combo[0], combo[2]], z[combo[1], combo[3]]])
        )

    edge_constr = opt_model.addVar(vtype=gp.GRB.INTEGER, name="edge_blk_cnt")
    cloud_constr = opt_model.addVar(vtype=gp.GRB.INTEGER, name="cld_blk_cnt")

    # floor with series of constraints
    opt_model.addConstr(
        edge_constr
        >= (
            gp.quicksum(
                [
                    z[combo]
                    for combo in z.keys()
                    if EDGE_NAME in combo
                    and "input" not in combo
                    and "output" not in combo
                ]
            )
            / max_layers_in_block
        ),
        name="edge_constr_lower_bound",
    )
    opt_model.addConstr(
        edge_constr + 1e-3
        <= (
            gp.quicksum(
                [
                    z[combo]
                    for combo in z.keys()
                    if EDGE_NAME in combo
                    and "input" not in combo
                    and "output" not in combo
                ]
            )
            / max_layers_in_block
        )
        + 1,
        name="edge_constr_upper_bound",
    )

    opt_model.addConstr(
        cloud_constr
        >= (
            gp.quicksum(
                [
                    z[combo]
                    for combo in z.keys()
                    if WISCAD_NAME in combo
                    and "input" not in combo
                    and "output" not in combo
                ]
            )
            / max_layers_in_block
        ),
        name="cloud_constr_lower_bound",
    )
    opt_model.addConstr(
        cloud_constr + 1e-3
        <= (
            gp.quicksum(
                [
                    z[combo]
                    for combo in z.keys()
                    if WISCAD_NAME in combo
                    and "input" not in combo
                    and "output" not in combo
                ]
            )
            / max_layers_in_block
        )
        + 1,
        name="cloud_constr_upper_bound",
    )

    # constrain number of bundles based on edge_constr
    opt_model.addConstr(
        edge_constr >=
        # number of bundles
        gp.quicksum(
            [
                y[combo]
                for combo in y.keys()
                if EDGE_NAME in combo and "input" not in combo and "output" not in combo
            ]
        ),
    )
    opt_model.addConstr(
        cloud_constr >=
        # number of bundles
        gp.quicksum(
            [
                y[combo]
                for combo in y.keys()
                if WISCAD_NAME in combo
                and "input" not in combo
                and "output" not in combo
            ]
        ),
    )

    def print_report(
        y, c, comp_latencies, comm_latencies, comp_energy_dict, comm_energy_dict
    ):
        if not opt_model.SolCount > 0:
            print("No solution")
            return
        for var in y:
            opt_model.setParam(gp.GRB.Param.SolutionNumber, 0)
            if y[var].xn > 0:
                print(f"\t{var}: {y[var].xn}")

        comp_latency = (y.prod(comp_latencies)).getValue()
        comm_latency = (c.prod(comm_latencies)).getValue()

        comp_energy = (y.prod(comp_energy_dict)).getValue()
        comm_energy = (c.prod(comm_energy_dict)).getValue()

        print(
            f"\t latency: {comp_latency + comm_latency} comp latency: {comp_latency} comm_latency:  {comm_latency}"
        )
        print(
            f"\t energy: {comp_energy + comm_energy} comp energy: {comp_energy} comm energy: {comm_energy}"
        )

        return (comp_latency, comm_latency, comp_energy, comm_energy)

    # minimize latency objective
    print("minimize latency")
    opt_model.setObjective(y.prod(comp_latencies) + c.prod(comm_latencies))

    opt_model.optimize()
    print_report(
        y, c, comp_latencies, comm_latencies, comp_energy_dict, comm_energy_dict
    )

    print(edge_constr.getAttr("x"))
    print(cloud_constr.getAttr("x"))
    # reset objective
    opt_model.setObjective(0.0)

    # minimize energy objective
    print("minimize energy")
    opt_model.setObjective(y.prod(comp_energy_dict) + c.prod(comm_energy_dict))
    opt_model.optimize()
    _, _, comm_energy, comp_energy = print_report(
        y, c, comp_latencies, comm_latencies, comp_energy_dict, comm_energy_dict
    )
    min_energy = comm_energy + comp_energy

    # find the max energy to bound sweeps
    opt_model.setObjective(0.0)  # reset
    opt_model.setObjective(
        y.prod(comp_energy_dict) + c.prod(comm_energy_dict), GRB.MAXIMIZE
    )
    opt_model.optimize()
    max_energy = opt_model.ObjVal

    # minimize latency, energy constr
    energy_constr_vals = np.linspace(min_energy, max_energy, num=20)
    lat_vals = []
    energy_vals = []
    freq_vals = []
    edge_bundles = []
    for energy_constr_val in energy_constr_vals:
        opt_model.setObjective(0.0)
        opt_model.setObjective(
            y.prod(comp_latencies) + c.prod(comm_latencies), GRB.MINIMIZE
        )
        energy_constr = opt_model.addConstr(
            gp.quicksum([y.prod(comp_energy_dict), c.prod(comm_energy_dict)])
            <= energy_constr_val
        )
        opt_model.optimize()
        if opt_model.SolCount > 0:
            comp_latency = (y.prod(comp_latencies)).getValue()
            comm_latency = (c.prod(comm_latencies)).getValue()
            lat_vals.append(comm_latency + comp_latency)

            comp_energy = (y.prod(comp_energy_dict)).getValue()
            comm_energy = (c.prod(comm_energy_dict)).getValue()
            energy_vals.append(comp_energy + comm_energy)

            freq = None
            edge_bundle = None
            for var in y:
                opt_model.setParam(gp.GRB.Param.SolutionNumber, 0)
                if y[var].xn > 0:
                    if EDGE_NAME in var:
                        freq = var[1]
                        edge_bundle = var[0]
            freq_vals.append(freq)
            edge_bundles.append(edge_bundle)

        opt_model.remove(energy_constr)
    print("\n\n Min Latency with Energy Constraint")
    min_lat_df = pd.DataFrame(
        {
            "lat": lat_vals,
            "energy": energy_vals,
            "energy_constr": list(energy_constr_vals),
            "frequency_setting": freq_vals,
            "conn": conn_name,
            "edge_bundle": edge_bundles,
        }
    )
    save_path = f"alexnet_{EDGE_NAME}_min_lat_df.csv"
    min_lat_df.to_csv(
        save_path, mode="a", header=not path.exists(save_path), index=False
    )

    # latency constraint, minimize energy
    opt_model.setObjective(
        y.prod(comp_latencies) + c.prod(comm_latencies), GRB.MINIMIZE
    )
    opt_model.optimize()
    min_latency = opt_model.objVal

    opt_model.setObjective(
        y.prod(comp_latencies) + c.prod(comm_latencies), GRB.MAXIMIZE
    )
    opt_model.optimize()
    max_latency = opt_model.objVal

    lat_constr_vals = np.linspace(min_latency, max_latency, num=20)
    lat_vals = []
    energy_vals = []
    freq_vals = []
    edge_bundles = []
    for lat_constr_val in lat_constr_vals:
        opt_model.setObjective(0.0)
        opt_model.setObjective(
            y.prod(comp_energy_dict) + c.prod(comm_energy_dict), GRB.MINIMIZE
        )
        lat_constr = opt_model.addConstr(
            gp.quicksum([y.prod(comp_latencies), c.prod(comm_latencies)])
            <= lat_constr_val
        )
        opt_model.optimize()
        # opt_model.write("lat_constr.lp")
        if opt_model.SolCount > 0:
            comp_latency = (y.prod(comp_latencies)).getValue()
            comm_latency = (c.prod(comm_latencies)).getValue()
            lat_vals.append(comp_latency + comm_latency)

            comp_energy = (y.prod(comp_energy_dict)).getValue()
            comm_energy = (c.prod(comm_energy_dict)).getValue()
            energy_vals.append(comp_energy + comm_energy)

            freq = None
            edge_bundle = None
            for var in y:
                opt_model.setParam(gp.GRB.Param.SolutionNumber, 0)
                if y[var].xn > 0:
                    if EDGE_NAME in var:
                        freq = var[1]
                        edge_bundle = var[0]
            freq_vals.append(freq)
            edge_bundles.append(edge_bundle)

        opt_model.remove(lat_constr)
    print("\n\n Min Energy with Latency Constraint")
    min_energy_df = pd.DataFrame(
        {
            "lat": lat_vals,
            "lat_constr": list(lat_constr_vals),
            "energy": energy_vals,
            "frequency_setting": freq_vals,
            "conn": conn_name,
            "edge_bundle": edge_bundles,
        }
    )
    save_path = f"alexnet_{EDGE_NAME}_min_energy_df.csv"
    min_energy_df.to_csv(
        save_path, mode="a", header=not path.exists(save_path), index=False
    )

    # minimize energy s.t. privacy constr
    # print("minimize energy s.t. private layers")
    # layers = [
    #     "conv1",
    #     "conv2",
    #     "conv3",
    #     "conv4",
    #     "conv5",
    #     "fc1",
    #     "fc2",
    #     "fc3",
    # ]
    # lat_vals = []
    # energy_vals = []
    # freq_vals = []
    # edge_bundles = []

    # for num_layers in range(1, len(layers) + 1):
    #     opt_model.setObjective(0.0)
    #     # minimize energy
    #     opt_model.setObjective(y.prod(comp_energy_dict) + c.prod(comm_energy_dict))
    #     private_layer_constrs = []
    #     for i in range(num_layers):
    #         private_layer_constrs.append(
    #             opt_model.addConstr(z[layers[i], EDGE_NAME] == 1)
    #         )

    #     opt_model.optimize()

    #     comp_latency = (y.prod(comp_latencies)).getValue()
    #     comm_latency = (c.prod(comm_latencies)).getValue()
    #     lat_vals.append(comp_latency + comm_latency)

    #     comp_energy = (y.prod(comp_energy_dict)).getValue()
    #     comm_energy = (c.prod(comm_energy_dict)).getValue()
    #     energy_vals.append(comp_energy + comm_energy)

    #     freq = None
    #     edge_bundle = None
    #     for var in y:
    #         opt_model.setParam(gp.GRB.Param.SolutionNumber, 0)
    #         if y[var].xn > 0:
    #             if EDGE_NAME in var:
    #                 freq = var[1]
    #                 edge_bundle = var[0]
    #     freq_vals.append(freq)
    #     edge_bundles.append(edge_bundle)

    #     for constr in private_layer_constrs:
    #         opt_model.remove(constr)

    # privacy_df = pd.DataFrame(
    #     {
    #         "lat": lat_vals,
    #         "energy": energy_vals,
    #         "frequency_setting": freq_vals,
    #         "conn": conn_name,
    #         "edge_bundle": edge_bundles,
    #         "num_private_layers": list(range(1, len(layers) + 1)),
    #     }
    # )
    # save_path = f"alexnet_{EDGE_NAME}_privacy_df.csv"
    # privacy_df.to_csv(
    #     save_path, mode="a", header=not path.exists(save_path), index=False
    # )

    def constrain_frequency(frequency: int):
        unused_freq_constrs = []
        for var in y:
            if var[2] == EDGE_NAME and var[1] != frequency:
                freq_constr = opt_model.addConstr(y[var] == 0)
                unused_freq_constrs.append(freq_constr)
        return unused_freq_constrs

    # generate solution if only fast frequency avail
    opt_model.setObjective(0.0)

    # minimize latency with fast frequency constraint
    def min_lat_energy_constraint(energy_constr_val):
        opt_model.setObjective(0.0)
        opt_model.setObjective(
            y.prod(comp_latencies) + c.prod(comm_latencies), GRB.MINIMIZE
        )

        energy_constr = opt_model.addConstr(
            gp.quicksum([y.prod(comp_energy_dict), c.prod(comm_energy_dict)])
            <= energy_constr_val
        )
        opt_model.optimize()
        print_report(
            y, c, comp_latencies, comm_latencies, comp_energy_dict, comm_energy_dict
        )

        opt_model.remove(energy_constr)

    def min_energy_lat_constraint(lat_constr_val):
        opt_model.setObjective(0.0)
        opt_model.setObjective(
            y.prod(comp_energy_dict) + c.prod(comm_energy_dict), GRB.MINIMIZE
        )
        lat_constr = opt_model.addConstr(
            gp.quicksum([y.prod(comp_latencies), c.prod(comm_latencies)])
            <= lat_constr_val
        )
        opt_model.optimize()

        print_report(
            y, c, comp_latencies, comm_latencies, comp_energy_dict, comm_energy_dict
        )
        opt_model.remove(lat_constr)

    def min_energy_privacy_constraint(num_layers):
        opt_model.setObjective(0.0)
        opt_model.setObjective(
            y.prod(comp_energy_dict) + c.prod(comm_energy_dict), GRB.MINIMIZE
        )

        private_layer_constrs = []
        for i in range(num_layers):
            private_layer_constrs.append(
                opt_model.addConstr(z[layers[i], EDGE_NAME] == 1)
            )

        opt_model.optimize()

        print_report(
            y, c, comp_latencies, comm_latencies, comp_energy_dict, comm_energy_dict
        )
        opt_model.remove(lat_constr)

        for constr in private_layer_constrs:
            opt_model.remove(constr)

    test_energy_constr = 95000
    print("\n\nstart comparison")
    min_lat_energy_constraint(test_energy_constr)
    min_energy_lat_constraint(0.1)
    # min_energy_privacy_constraint(3)
    # min_energy_privacy_constraint(5)

    print("max freq")
    max_frequency = 921600000 if USING_NANO else 1200000
    unused_fast_constrs = constrain_frequency(max_frequency)
    min_lat_energy_constraint(test_energy_constr)
    min_energy_lat_constraint(0.1)
    # min_energy_privacy_constraint(3)
    # min_energy_privacy_constraint(5)
    for constr in unused_fast_constrs:
        opt_model.remove(constr)

    print("min freq")
    min_frequency = 537600000
    unused_slow_constrs = constrain_frequency(min_frequency)
    min_lat_energy_constraint(test_energy_constr)
    min_energy_lat_constraint(0.1)
    # min_energy_privacy_constraint(3)
    # min_energy_privacy_constraint(5)
    for constr in unused_slow_constrs:
        opt_model.remove(constr)
    print("\n\n")

    opt_model.setObjective(0.0)
    opt_model.setObjective(
        y.prod(comp_latencies) + c.prod(comm_latencies), GRB.MINIMIZE
    )
    opt_model.optimize()
    comm_latency, comp_latency, comp_energy, comm_energy = print_report(
        y, c, comp_latencies, comm_latencies, comp_energy_dict, comm_energy_dict
    )

    df = pd.DataFrame(
        {
            "lat": [comm_latency + comp_latency],
            "energy": [comm_energy + comp_energy],
            "conn": conn_name,
        }
    )
    save_path = f"alexnet_{EDGE_NAME}_min_lat_fast.csv"
    df.to_csv(save_path, mode="a", header=not path.exists(save_path), index=False)

    # minimize energy with fast frequency constraint
    opt_model.setObjective(0.0)
    opt_model.setObjective(
        y.prod(comp_energy_dict) + c.prod(comm_energy_dict), GRB.MINIMIZE
    )
    opt_model.optimize()
    comm_latency, comp_latency, comp_energy, comm_energy = print_report(
        y, c, comp_latencies, comm_latencies, comp_energy_dict, comm_energy_dict
    )

    df = pd.DataFrame(
        {
            "lat": [comm_latency + comp_latency],
            "energy": [comm_energy + comp_energy],
            "conn": conn_name,
        }
    )
    save_path = f"alexnet_{EDGE_NAME}_min_energy_fast.csv"
    df.to_csv(save_path, mode="a", header=not path.exists(save_path), index=False)

    # lat_vals = []
    # energy_vals = []
    # freq_vals = []
    # edge_bundles = []

    # for num_layers in range(1, len(layers) + 1):
    #     opt_model.setObjective(0.0)
    #     # minimize energy
    #     opt_model.setObjective(y.prod(comp_energy_dict) + c.prod(comm_energy_dict))
    #     private_layer_constrs = []
    #     for i in range(num_layers):
    #         private_layer_constrs.append(
    #             opt_model.addConstr(z[layers[i], EDGE_NAME] == 1)
    #         )

    #     opt_model.optimize()

    #     comp_latency = (y.prod(comp_latencies)).getValue()
    #     comm_latency = (c.prod(comm_latencies)).getValue()
    #     lat_vals.append(comp_latency + comm_latency)

    #     comp_energy = (y.prod(comp_energy_dict)).getValue()
    #     comm_energy = (c.prod(comm_energy_dict)).getValue()
    #     energy_vals.append(comp_energy + comm_energy)

    #     freq = None
    #     edge_bundle = None
    #     for var in y:
    #         opt_model.setParam(gp.GRB.Param.SolutionNumber, 0)
    #         if y[var].xn > 0:
    #             if EDGE_NAME in var:
    #                 freq = var[1]
    #                 edge_bundle = var[0]
    #     freq_vals.append(freq)
    #     edge_bundles.append(edge_bundle)

    #     for constr in private_layer_constrs:
    #         opt_model.remove(constr)

    # privacy_df = pd.DataFrame(
    #     {
    #         "lat": lat_vals,
    #         "energy": energy_vals,
    #         "frequency_setting": freq_vals,
    #         "conn": conn_name,
    #         "edge_bundle": edge_bundles,
    #         "num_private_layers": list(range(1, len(layers) + 1)),
    #     }
    # )
    # save_path = f"alexnet_{EDGE_NAME}_privacy_fast.csv"
    # privacy_df.to_csv(
    #     save_path, mode="a", header=not path.exists(save_path), index=False
    # )

    for soln in range(opt_model.getAttr(gp.GRB.Attr.SolCount)):
        print(f"solution {soln}")
        for var in y:
            opt_model.setParam(gp.GRB.Param.SolutionNumber, soln)
            if y[var].xn > 0:
                print(f"\t{var}: {y[var].xn}")

    with open("alexnet_experiments.txt", "a") as log:
        log.write(f"max block size: {max_layers_in_block}\n")
        log.write(f"bandwidth: {bandwidth}\n")
        for soln in range(opt_model.getAttr(gp.GRB.Attr.SolCount)):
            opt_model.setParam(gp.GRB.Param.SolutionNumber, soln)

            log.write(f"\tsolution {soln}\n")
            log.write(f"\tsolution latency: {opt_model.poolObjVal}\n")

            for var in y:
                if y[var].xn > 0:
                    log.write(f"\t\t{var}: {y[var].xn}\n")


if __name__ == "__main__":
    for conn_name, bandwidth in [
        ("3G", (2.0275 / 8) * 1e6),
        ("4G", (13.76 / 8) * 1e6),
        ("WiFi", (54.97 / 8) * 1e6),
    ]:
        for max_blocks in range(8, 9):
            run_energy_optimization(bandwidth, max_blocks, conn_name)
