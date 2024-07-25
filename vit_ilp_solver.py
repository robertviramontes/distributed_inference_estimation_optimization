import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from os import path

from segmented_vit_profile import get_input_shape_abbv
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


def run_energy_optimization(bandwidth: float, max_layers_in_block: int, conn_name=None):
    # load in profiles
    USING_NANO = True
    EDGE_NAME = NANO_NAME if USING_NANO else LEPOTATO_NAME

    lepotato = pd.read_parquet("nano_vit_profiles_fixed/latency_and_energy.parquet")
    lepotato["num_blocks"] = lepotato["layers"].apply(lambda x: len(x.split("__")))
    lepotato["layers"] = lepotato["layers"].apply(lambda x: x.replace("encoder", "enc"))
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

    wiscad = pd.read_csv("wiscad_vit_segments_latencies.csv", index_col=[0])
    wiscad["durs"] = wiscad["durs"]
    wiscad["num_blocks"] = wiscad["layers"].apply(lambda x: len(x.split("__")))
    wiscad["layers"] = wiscad["layers"].apply(lambda x: x.replace("encoder", "enc"))
    wiscad_vals = wiscad[wiscad["num_blocks"] <= max_layers_in_block]

    wiscad_vals = (
        wiscad[["layers", "durs"]]
        .groupby(["layers"])
        .apply(trimmean, 0.2)
    )

    computation_dict = {}
    assignment_list = []
    layers = []

    combo_bases = [
        "conv_proj",
        "process_input_operations",
        "add_pos_embedding",
        "enc_layer_0",
        "enc_layer_1",
        "enc_layer_2",
        "enc_layer_3",
        "enc_layer_4",
        "enc_layer_5",
        "enc_layer_6",
        "enc_layer_7",
        "enc_layer_8",
        "enc_layer_9",
        "enc_layer_10",
        "enc_layer_11",
        "head",
    ]

    for _, row in lepotato_latencies.iterrows():
        layer_name = row["layers"]
        frequency = row["frequency"]
        latency = row["durs"]
        computation_dict[(layer_name, frequency, EDGE_NAME)] = latency
        if layer_name in combo_bases:
            if not ((layer_name, EDGE_NAME) in assignment_list):
                assignment_list.append((layer_name, EDGE_NAME))
            if not (layer_name in layers):
                layers.append(layer_name)
        del layer_name, frequency, latency

    for layer_name, latency in wiscad_vals.items():
        frequency = -1
        computation_dict[(layer_name, frequency, WISCAD_NAME)] = latency
        if layer_name in combo_bases:
            assignment_list.append((layer_name, WISCAD_NAME))

    blocks_to_consider = list(wiscad_vals.index)

    opt_model = gp.Model("ViT")

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
            comm_combo[1], conn_name, comm_combo[2], get_input_shape_abbv
        )
        comm_energy_dict[comm_combo] = estimate_communication_energy_asymmetric(
            comm_combo[1], conn_name, comm_combo[2], get_input_shape_abbv
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
    save_path = f"_vit_{EDGE_NAME}_min_lat_df.csv"
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
    save_path = f"_vit_{EDGE_NAME}_min_energy_df.csv"
    min_energy_df.to_csv(
        save_path, mode="a", header=not path.exists(save_path), index=False
    )

    # # minimize energy s.t. privacy constr
    # print("minimize energy s.t. private layers")
    # layers = [
    #     "conv1",
    #     "conv2",
    #     "conv3",
    #     "conv4",
    #     "conv5",
    #     "conv6",
    #     "conv7",
    #     "conv8",
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
    # save_path = f"_vgg11_{EDGE_NAME}_privacy_df.csv"
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

        print_report(y, c, comp_latencies, comm_latencies, comp_energy_dict, comm_energy_dict)
        opt_model.remove(lat_constr)

        for constr in private_layer_constrs:
            opt_model.remove(constr)

    print("\n\nstart comparison")
    test_lat_constraint = 0.500
    test_energy_constraint = 1_800_000
    min_lat_energy_constraint(test_energy_constraint)
    min_energy_lat_constraint(test_lat_constraint)
    #min_energy_privacy_constraint(5)
    #min_energy_privacy_constraint(7)

    print('\033[94m' + "max freq" + '\033[0m')
    max_frequency = 921600000 if USING_NANO else 1200000
    unused_fast_constrs = constrain_frequency(max_frequency)
    min_lat_energy_constraint(test_energy_constraint)
    min_energy_lat_constraint(test_lat_constraint)
    #min_energy_privacy_constraint(5)
    #min_energy_privacy_constraint(7)
    for constr in unused_fast_constrs:
        opt_model.remove(constr)

    print('\033[94m' + "min freq" + '\033[0m')
    min_frequency = 614400000
    unused_slow_constrs = constrain_frequency(min_frequency)
    min_lat_energy_constraint(test_energy_constraint)
    min_energy_lat_constraint(test_lat_constraint)
    #min_energy_privacy_constraint(5)
    #min_energy_privacy_constraint(7)
    for constr in unused_slow_constrs:
        opt_model.remove(constr)


if __name__ == "__main__":
    # for bandwidth in [1e5]:
    #     for max_blocks in range(11, 12):
    #         run_optimization(bandwidth, max_blocks)
    for conn_name, bandwidth in [
        # ("3G", (2.0275 / 8) * 1e6),
        ("4G", (13.76 / 8) * 1e6),
        # ("WiFi", (54.97 / 8) * 1e6),
    ]:
        for max_blocks in range(18, 19):
            run_energy_optimization(bandwidth, max_blocks, conn_name)
