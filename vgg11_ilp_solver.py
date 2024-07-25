import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from os import path

from segmented_vgg11_profile import get_input_shape
from ilp_solver_common import (
    trimmean,
    potato_trimmean,
    estimate_communication_latency,
    estimate_communication_energy_asymmetric,
    estimate_communication_latency_asymmetric,
    LEPOTATO_NAME,
    NANO_NAME,
    WISCAD_NAME,
)

USE_NANO = True


def run_optimization(bandwidth: float, max_layers_in_block):
    # load in latency values
    lepotato = pd.read_csv("lepotato_vgg11_segments_latencies.csv", index_col=[0])
    lepotato["num_blocks"] = lepotato["layers"].apply(lambda x: len(x.split("_")))
    lepotato_vals = lepotato[lepotato["num_blocks"] <= max_layers_in_block]
    lepotato_vals = (
        lepotato_vals.drop(columns=["num_blocks"])
        .groupby(["layers"])
        .apply(trimmean, 0.2)
    )

    wiscad = pd.read_csv("wiscad_vgg11_segments_latencies.csv", index_col=[0])
    wiscad["durs"] = wiscad["durs"] / 1000
    wiscad["num_blocks"] = wiscad["layers"].apply(lambda x: len(x.split("_")))
    wiscad_vals = wiscad[wiscad["num_blocks"] <= max_layers_in_block]
    wiscad_vals = (
        wiscad_vals.drop(columns=["num_blocks"])
        .groupby(["layers"])
        .apply(trimmean, 0.2)
    )

    nano = pd.read_csv("nano_vgg11_segments_latencies.csv", index_col=[0])
    nano["durs"] = nano["durs"] / 1000
    nano["num_blocks"] = nano["layers"].apply(lambda x: len(x.split("_")))
    nano_vals = nano[nano["num_blocks"] <= max_layers_in_block]
    nano_vals = (
        nano_vals.drop(columns=["num_blocks"]).groupby(["layers"]).apply(trimmean, 0.2)
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

    if USE_NANO:
        for layer_name, latency in nano_vals.items():
            computation_dict[(layer_name, NANO_NAME)] = latency
            if len(layer_name.split("_")) == 1:
                assignment_list.append((layer_name, NANO_NAME))

    blocks_to_consider = list(wiscad_vals.index)

    opt_model = gp.Model("VGG11")

    # computation times
    computations, comp_latencies = gp.multidict(computation_dict)

    layer_device = gp.tuplelist(
        [("input", LEPOTATO_NAME)] + assignment_list + [("output", LEPOTATO_NAME)]
    )

    # generate communication combinations
    comm_combos = []
    for num_on_edge, layer in enumerate(layer_device):
        if num_on_edge == len(layer_device) - 1:
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
            comm_combo[1],
            # bandwidth
            (bandwidth if WISCAD_NAME in comm_combo else (1.375 * 1e5)),
            get_input_shape,
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
    for num_on_edge, layer in enumerate(layers):
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
        if USE_NANO:
            opt_model.addConstr(
                z[(layer, NANO_NAME)]
                == gp.quicksum(y[block, NANO_NAME] for block in blocks_containing_layer)
            )

    # constr to solve the communication var
    for combo in comm_combos:
        opt_model.addConstr(
            c[combo] == gp.and_([z[combo[0], combo[2]], z[combo[1], combo[3]]])
        )

    edge_constr = opt_model.addVar(vtype=gp.GRB.INTEGER, name="edge_blk_cnt")
    cloud_constr = opt_model.addVar(vtype=gp.GRB.INTEGER, name="cld_blk_cnt")
    hub_constr = opt_model.addVar(vtype=gp.GRB.INTEGER, name="hub_blk_cnt")

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
    if USE_NANO:
        opt_model.addConstr(
            hub_constr
            >= (
                gp.quicksum(
                    [
                        z[combo]
                        for combo in z.keys()
                        if NANO_NAME in combo
                        and "input" not in combo
                        and "output" not in combo
                    ]
                )
                / max_layers_in_block
            ),
            name="hub_constr_lower_bound",
        )
        opt_model.addConstr(
            hub_constr + 1e-3
            <= (
                gp.quicksum(
                    [
                        z[combo]
                        for combo in z.keys()
                        if NANO_NAME in combo
                        and "input" not in combo
                        and "output" not in combo
                    ]
                )
                / max_layers_in_block
            )
            + 1,
            name="hub_constr_upper_bound",
        )

        opt_model.addConstr(
            hub_constr
            # max_layers_per_device
            >= gp.quicksum(
                [
                    y[combo]
                    for combo in y.keys()
                    if NANO_NAME in combo
                    and "input" not in combo
                    and "output" not in combo
                ]
            )
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
        cloud_constr
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

    with open("temp.txt", "a") as log:
        log.write(f"max block size: {max_layers_in_block}\n")
        log.write(f"bandwidth: {bandwidth}\n")
        for soln in range(opt_model.getAttr(gp.GRB.Attr.SolCount)):
            opt_model.setParam(gp.GRB.Param.SolutionNumber, soln)

            log.write(f"\tsolution {soln}\n")
            log.write(f"\tsolution latency: {opt_model.poolObjVal}\n")

            for var in y:
                if y[var].xn > 0:
                    log.write(f"\t\t{var}: {y[var].xn}\n")

    if not USE_NANO and max_layers_in_block == 11:
        all_combo = "conv1_conv2_conv3_conv4_conv5_conv6_conv7_conv8_fc1_fc2_fc3"
        all_layers = all_combo.split("_")

        for num_on_edge in range(len(all_layers) + 1):
            on_edge = all_layers[:num_on_edge]
            on_cloud = all_layers[num_on_edge:]
            edge_block = "_".join(on_edge)
            cloud_block = "_".join(on_cloud)

            edge_comp_latency = (
                0 if edge_block == "" else comp_latencies[edge_block, LEPOTATO_NAME]
            )
            cloud_comp_latency = (
                0 if cloud_block == "" else comp_latencies[cloud_block, WISCAD_NAME]
            )
            result_comm_latency = (
                0
                if not ("fc3" in cloud_block)
                else comm_latencies["fc3", "output", WISCAD_NAME, LEPOTATO_NAME]
            )
            transition_comm_latency = (
                0
                if cloud_block == ""
                else (
                    comm_latencies["input", on_cloud[0], LEPOTATO_NAME, WISCAD_NAME]
                    if edge_block == ""
                    else comm_latencies[
                        on_edge[-1], on_cloud[0], LEPOTATO_NAME, WISCAD_NAME
                    ]
                )
            )

            latency = (
                edge_comp_latency
                + cloud_comp_latency
                + result_comm_latency
                + transition_comm_latency
            )
            print(
                f"{len(on_edge)} on edge, {len(on_cloud)} on cloud: latency {latency}"
            )

    if USE_NANO and max_layers_in_block == 11:
        all_combo = "conv1_conv2_conv3_conv4_conv5_conv6_conv7_conv8_fc1_fc2_fc3"
        all_layers = all_combo.split("_")
        latencies = []
        for num_on_edge in range(len(all_layers) + 1):
            for num_on_hub in range(len(all_layers) + 1 - num_on_edge):
                on_edge = all_layers[:num_on_edge]
                on_hub = all_layers[num_on_edge : num_on_edge + num_on_hub]
                on_cloud = all_layers[num_on_edge + num_on_hub :]
                edge_block = "_".join(on_edge)
                hub_block = "_".join(on_hub)
                cloud_block = "_".join(on_cloud)

                edge_comp_latency = (
                    0 if edge_block == "" else comp_latencies[edge_block, LEPOTATO_NAME]
                )
                hub_comp_latency = (
                    0 if hub_block == "" else comp_latencies[hub_block, NANO_NAME]
                )
                cloud_comp_latency = (
                    0 if cloud_block == "" else comp_latencies[cloud_block, WISCAD_NAME]
                )
                result_comm_latency = (
                    comm_latencies["fc3", "output", WISCAD_NAME, LEPOTATO_NAME]
                    if ("fc3" in cloud_block)
                    else (
                        comm_latencies["fc3", "output", NANO_NAME, LEPOTATO_NAME]
                        if "fc3" in hub_block
                        else 0
                    )
                )

                if hub_block == "":
                    hub_transition_comm_latency = 0
                else:
                    first_on_hub = on_hub[0]
                    layer_num = all_layers.index(first_on_hub)
                    if layer_num == 0:
                        hub_transition_comm_latency = comm_latencies[
                            "input", first_on_hub, LEPOTATO_NAME, NANO_NAME
                        ]
                    else:
                        prev_layer = all_layers[layer_num - 1]
                        if prev_layer in edge_block:
                            hub_transition_comm_latency = comm_latencies[
                                prev_layer, first_on_hub, LEPOTATO_NAME, NANO_NAME
                            ]
                        elif prev_layer in cloud_block:
                            hub_transition_comm_latency = comm_latencies[
                                prev_layer, first_on_hub, WISCAD_NAME, NANO_NAME
                            ]
                        else:
                            raise ValueError("uh oh")

                if cloud_block == "":
                    cloud_transition_comm_latency = 0
                else:
                    first_on_cloud = on_cloud[0]
                    layer_num = all_layers.index(first_on_cloud)
                    if layer_num == 0:
                        cloud_transition_comm_latency = comm_latencies[
                            "input", first_on_cloud, LEPOTATO_NAME, WISCAD_NAME
                        ]
                    else:
                        prev_layer = all_layers[layer_num - 1]
                        if prev_layer in edge_block:
                            cloud_transition_comm_latency = comm_latencies[
                                prev_layer, first_on_cloud, LEPOTATO_NAME, WISCAD_NAME
                            ]
                        elif prev_layer in hub_block:
                            cloud_transition_comm_latency = comm_latencies[
                                prev_layer, first_on_cloud, NANO_NAME, WISCAD_NAME
                            ]
                        else:
                            raise ValueError("uh oh 2")

                latency = (
                    edge_comp_latency
                    + hub_comp_latency
                    + cloud_comp_latency
                    + result_comm_latency
                    + hub_transition_comm_latency
                    + cloud_transition_comm_latency
                )
                latencies.append((len(on_edge), len(on_hub), len(on_cloud), latency))

        latencies = sorted(latencies, key=lambda tuple: tuple[-1])
        print(latencies[0])

    if max_layers_in_block == 11:
        # generate an pedram model

        pedram_model = opt_model.copy()
        pedram_model.remove(pedram_model.getConstrByName("edge_constr_lower_bound"))
        pedram_model.remove(pedram_model.getConstrByName("edge_constr_upper_bound"))
        pedram_model.remove(pedram_model.getConstrByName("cloud_constr_lower_bound"))
        pedram_model.remove(pedram_model.getConstrByName("cloud_constr_upper_bound"))

        if USE_NANO:
            pedram_model.remove(pedram_model.getConstrByName("hub_constr_lower_bound"))
            pedram_model.remove(pedram_model.getConstrByName("hub_constr_upper_bound"))

        pedram_model.optimize()

        for var in pedram_model.getVars():
            if var.x > 0.1:
                print(f"{var}: {var.x}")
        print(pedram_model.objVal)


def run_energy_optimization(bandwidth: float, max_layers_in_block: int, conn_name=None):
    # load in profiles
    USING_NANO = True
    EDGE_NAME = NANO_NAME if USING_NANO else LEPOTATO_NAME

    lepotato = pd.read_parquet("nano_vgg11_profiles_fixed2/latency_and_energy.parquet")
    lepotato["num_blocks"] = lepotato["layers"].apply(lambda x: len(x.split("_")))
    if not USING_NANO:
        lepotato["durs"] = lepotato["durs"] / 1e9

    # filter to only the bundle sizes of interest
    lepotato_vals = lepotato[lepotato["num_blocks"] <= max_layers_in_block]

    lepotato_vals = lepotato_vals.drop(columns=["num_blocks"]).groupby(
        ["layers", "frequency"], as_index=False
    )

    if not USING_NANO:
        lepotato_latencies = lepotato_vals.apply(potato_trimmean, 0.2)
    else:
        lepotato_latencies = lepotato_vals["durs"].apply(trimmean, 0.2)
    lepotato_freqs = lepotato["frequency"].unique()

    wiscad = pd.read_csv("wiscad_vgg11_segments_latencies.csv", index_col=[0])
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

    opt_model = gp.Model("VGG11")

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

    # edge constr >= number of bundles assigned to edge
    opt_model.addConstr(
        edge_constr
        >= gp.quicksum(
            [
                y[combo]
                for combo in y.keys()
                if EDGE_NAME in combo and "input" not in combo and "output" not in combo
            ]
        )
    )
    opt_model.addConstr(
        cloud_constr
        # number of bundles
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
    opt_model.write("out.lp")

    opt_model.optimize()
    print_report(
        y, c, comp_latencies, comm_latencies, comp_energy_dict, comm_energy_dict
    )

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
    save_path = f"_vgg11_{EDGE_NAME}_min_lat_df.csv"
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
    save_path = f"_vgg11_{EDGE_NAME}_min_energy_df.csv"
    min_energy_df.to_csv(
        save_path, mode="a", header=not path.exists(save_path), index=False
    )

    # minimize energy s.t. privacy constr
    print("minimize energy s.t. private layers")
    layers = [
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
    lat_vals = []
    energy_vals = []
    freq_vals = []
    edge_bundles = []
    for num_layers in range(1, len(layers) + 1):
        opt_model.setObjective(0.0)
        # minimize energy
        opt_model.setObjective(y.prod(comp_energy_dict) + c.prod(comm_energy_dict))
        private_layer_constrs = []
        for i in range(num_layers):
            private_layer_constrs.append(
                opt_model.addConstr(z[layers[i], EDGE_NAME] == 1)
            )

        opt_model.optimize()

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

        for constr in private_layer_constrs:
            opt_model.remove(constr)

    privacy_df = pd.DataFrame(
        {
            "lat": lat_vals,
            "energy": energy_vals,
            "frequency_setting": freq_vals,
            "conn": conn_name,
            "edge_bundle": edge_bundles,
            "num_private_layers": list(range(1, len(layers) + 1)),
        }
    )
    save_path = f"_vgg11_{EDGE_NAME}_privacy_df.csv"
    privacy_df.to_csv(
        save_path, mode="a", header=not path.exists(save_path), index=False
    )

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

        print_report(y, c, comp_latencies, comm_latencies, comp_energy_dict, comm_energy_dict)
        opt_model.remove(lat_constr)

        for constr in private_layer_constrs:
            opt_model.remove(constr)

    print("\n\nstart comparison")
    test_lat_constraint = 0.16
    test_energy_constraint = 650_000
    min_lat_energy_constraint(test_energy_constraint)
    min_energy_lat_constraint(test_lat_constraint)
    # min_energy_privacy_constraint(5)
    # min_energy_privacy_constraint(7)

    print("max freq")
    max_frequency = 921600000 if USING_NANO else 1200000
    unused_fast_constrs = constrain_frequency(max_frequency)
    min_lat_energy_constraint(test_energy_constraint)
    min_energy_lat_constraint(test_lat_constraint)
    # min_energy_privacy_constraint(5)
    # min_energy_privacy_constraint(7)
    for constr in unused_fast_constrs:
        opt_model.remove(constr)

    print("min freq")
    min_frequency = 537600000
    unused_slow_constrs = constrain_frequency(min_frequency)
    min_lat_energy_constraint(test_energy_constraint)
    min_energy_lat_constraint(test_lat_constraint)
    # min_energy_privacy_constraint(5)
    # min_energy_privacy_constraint(7)
    for constr in unused_slow_constrs:
        opt_model.remove(constr)


if __name__ == "__main__":
    # for bandwidth in [1e5]:
    #     for max_blocks in range(11, 12):
    #         run_optimization(bandwidth, max_blocks)
    for conn_name, bandwidth in [
        # ("3G", (2.0275 / 8) * 1e6),
        # ("4G", (13.76 / 8) * 1e6),
        ("WiFi", (54.97 / 8) * 1e6),
    ]:
        for max_blocks in range(11, 12):
            run_energy_optimization(bandwidth, max_blocks, conn_name)
