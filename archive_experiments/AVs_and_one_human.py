import itertools
import os
import pandas as pd
import numpy as np
import random

from routerl import TrafficEnvironment


from routerl.keychain import Keychain as kc
from routerl.utilities import get_params

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.lines as lines
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch

# ============================================================
# Run selected joint actions with dynamic traffic-light rule
# ============================================================

threshold = 3
nb_agents = 15
tl_y = 5
cycle_length = 50

tl_0_below_threshold = 21
tl_1_below_threshold = cycle_length - 2 * tl_y - tl_0_below_threshold  # 19

tl_0_above_threshold = 9
tl_1_above_threshold = cycle_length - 2 * tl_y - tl_0_above_threshold  # 31


def id_to_strategy(id):
    s = [0 for _ in range(10)]
    i = 0
    while id > 0:
        if id % 2 == 1:
            s[9 - i] = 1
        id = id // 2
        i += 1
    return s

def update_trafficlight(t0,t1,ty,offset=0):

    '''
    t0 = time of green light for route 0, red light for route 1 \\
    t1 = time of red light for route 0, green light for route 1 \\
    ty = time of yellow light for both routes (symmetrical)

    This function updates the network file used for simulations. \\
    !!! Verify paths exist on your machine !!!
    '''

    # template file
    read_file_name = "../networks/two_route_trafficlight/two_route_trafficlight.net.xml"

    # path within the /networks/ folder
    network_file_name = "../networks/two_route_trafficlight/two_route_trafficlight.net.xml"

    tllogic = []
    tllogic.append( "\t<tlLogic id=\"J2\" type=\"static\" programID=\"0\" offset=\"%s\">\n"%(offset))
    tllogic.append( "\t\t<phase duration=\"%s\" state=\"Gr\"/>\n"%(t1))
    tllogic.append( "\t\t<phase duration=\"%s\"  state=\"yr\"/>\n"%(ty))
    tllogic.append( "\t\t<phase duration=\"%s\" state=\"rG\"/>\n"%(t0))
    tllogic.append( "\t\t<phase duration=\"%s\"  state=\"ry\"/>\n"%(ty))
    tllogic.append( "\t</tlLogic>\n")

    with open(read_file_name, "r") as fr:
        lines = fr.readlines()
        print(lines)

    with open(network_file_name, "w") as fw:
        fw.writelines(lines[:76])
        fw.writelines(tllogic)
        fw.writelines(lines[82:])
    
    return None

def trafficlight_from_joint_action(s):
    nb_1 = sum(s)

    if nb_1 < threshold:
        return tl_0_below_threshold, tl_1_below_threshold, tl_y
    else:
        return tl_0_above_threshold, tl_1_above_threshold, tl_y




def create_environment(records_folder, nb_agents=23, open_gui=False):
    sumo_type = "sumo-gui" if open_gui else "sumo"

    env = TrafficEnvironment(
        agent_parameters={
            "num_agents": nb_agents,
            "new_machines_after_mutation": 10,
            "machine_parameters": {
                "behavior": "selfish"
            }
        },
        simulator_parameters={
            "network_name": "two_route_trafficlight",
            "sumo_type": sumo_type,
            "custom_network_folder": "../networks/two_route_trafficlight",
        },
        path_generation_parameters={
            "origins": ["E0"],
            "destinations": ["E2"],
            "number_of_paths": 2
        },
        plotter_parameters={
            "records_folder": records_folder,
        }
    )

    return env



import shutil
from pathlib import Path


def int_to_strategy(idx, width):
    """
    Converts integer idx into a binary list of length width.
    """
    if idx < 0 or idx >= 2**width:
        raise ValueError(f"idx must be between 0 and {2**width - 1}")

    return [int(x) for x in format(idx, f"0{width}b")]


def joint_id_to_avs_and_human13(joint_action_id, nb_avs=10):
    """
    Joint action has 11 bits:

        first 10 bits -> AV actions
        last bit      -> human 13 action
    """

    bits = int_to_strategy(joint_action_id, nb_avs + 1)

    av_actions = bits[:nb_avs]
    human13_action = bits[nb_avs]

    return av_actions, human13_action


def _agent_sort_key(agent):
    try:
        return (0, int(agent.id))
    except Exception:
        return (1, str(agent.id))


def set_human_actions_only_human13_variable(
    env,
    human13_action,
    human13_id=13,
    fixed_human_action=0,
):
    """
    Human 13 receives the chosen action.
    All other humans are fixed to route 0.
    """

    env.human_agents = sorted(env.human_agents, key=_agent_sort_key)

    found_human13 = False

    for human in env.human_agents:
        if int(human.id) == int(human13_id):
            human.default_action = int(human13_action)
            found_human13 = True
        else:
            human.default_action = int(fixed_human_action)

        print(f"Human {human.id} default action is {human.default_action}")

    if not found_human13:
        available_ids = [human.id for human in env.human_agents]
        raise ValueError(
            f"Could not find human agent with id {human13_id}. "
            f"Available human ids are: {available_ids}"
        )


def move_single_episode_csv(tmp_records_folder, final_records_folder, episode_id):
    """
    Moves the generated ep*.csv from the temporary folder to:

        training_records/episodes/ep{episode_id}.csv

    Therefore, each final episode file corresponds exactly to one joint action:
        ep0.csv    -> joint_action_id 0
        ep1.csv    -> joint_action_id 1
        ...
        ep2047.csv -> joint_action_id 2047
    """

    tmp_records_folder = Path(tmp_records_folder)
    final_records_folder = Path(final_records_folder)

    final_episodes_folder = final_records_folder / "episodes"
    final_episodes_folder.mkdir(parents=True, exist_ok=True)

    episode_csvs = list(tmp_records_folder.rglob("ep*.csv"))

    if len(episode_csvs) == 0:
        raise FileNotFoundError(
            f"No ep*.csv file was generated in {tmp_records_folder}"
        )

    # Usually there should be exactly one. If there are several, use the newest.
    source_csv = max(episode_csvs, key=lambda p: p.stat().st_mtime)

    destination_csv = final_episodes_folder / f"ep{episode_id}.csv"

    if destination_csv.exists():
        raise FileExistsError(
            f"{destination_csv} already exists. "
            f"Delete it first or choose a different output folder."
        )

    shutil.move(str(source_csv), str(destination_csv))

    return destination_csv

def num_times_to_human_strategy(num_times, nb_humans=5):
    """
    num_times=1  -> [0, 0, 0, 0, 0]
    num_times=2  -> [0, 0, 0, 0, 1]
    ...
    num_times=32 -> [1, 1, 1, 1, 1]
    """

    idx = num_times - 1

    if idx < 0 or idx >= 2**nb_humans:
        raise ValueError(f"num_times must be between 1 and {2**nb_humans}")

    return [int(x) for x in format(idx, f"0{nb_humans}b")]

def simulate_avs_plus_human13(
    records_folder,
    av_actions,
    human13_action,
    nb_agents=15,
    open_gui=False,
    human13_id=13,
):
    """
    Runs one simulation where:
        - AVs use av_actions
        - human 13 uses human13_action
        - all other humans use route 0
    """

    if len(av_actions) != 10:
        raise ValueError(f"Expected 10 AV actions, got {len(av_actions)}")

    env = create_environment(
        records_folder=records_folder,
        nb_agents=nb_agents,
        open_gui=open_gui,
    )

    env.start()

    try:
        env.mutation()
        env.reset()

        env.human_agents = sorted(env.human_agents, key=_agent_sort_key)

        if hasattr(env, "machine_agents"):
            env.machine_agents = sorted(env.machine_agents, key=_agent_sort_key)

        print("env.human_agents:", [h.id for h in env.human_agents])
        print("env.machine_agents:", [m.id for m in env.machine_agents])
        print("AV actions:", av_actions)
        print("Human 13 action:", human13_action)
        print()

        set_human_actions_only_human13_variable(
            env,
            human13_action=human13_action,
            human13_id=human13_id,
            fixed_human_action=0,
        )

        for action in av_actions:
            env.step(int(action))

    finally:
        env.stop_simulation()

import re
import glob

def get_next_training_record_index(prefix="training_records"):
    folders = glob.glob(f"{prefix}*")
    indices = []

    for folder in folders:
        match = re.search(rf"{prefix}(\d+)$", folder)
        if match:
            indices.append(int(match.group(1)))

    if len(indices) == 0:
        return 1

    return max(indices) + 1


def prepare_tmp_records_folder_for_routerl(tmp_records_folder, final_records_folder):
    """
    routerl expects paths.csv to exist inside records_folder.
    Since each simulation uses a temporary records folder, copy paths.csv there.
    """

    tmp_records_folder = Path(tmp_records_folder).resolve()
    final_records_folder = Path(final_records_folder).resolve()

    tmp_records_folder.mkdir(parents=True, exist_ok=True)

    source_paths_csv = final_records_folder / "paths.csv"
    destination_paths_csv = tmp_records_folder / "paths.csv"

    if not source_paths_csv.exists():
        raise FileNotFoundError(
            f"Could not find source paths.csv:\n{source_paths_csv}\n\n"
            "Check that training_records/paths.csv exists and that you are running "
            "the script from the expected working directory."
        )

    shutil.copy2(source_paths_csv, destination_paths_csv)

    if not destination_paths_csv.exists():
        raise FileNotFoundError(
            f"Copy failed. Expected paths.csv here:\n{destination_paths_csv}"
        )

    print(f"Copied paths.csv to: {destination_paths_csv}")
    print(f"paths.csv exists before TrafficEnvironment: {destination_paths_csv.exists()}")

    return destination_paths_csv


def run_one_joint_action_with_human13_single_folder(
    joint_action_id,
    final_records_folder="training_records",
    offset=0,
    human13_id=13,
    count_human13_in_trafficlight=False,
):
    """
    Runs one of the 2048 joint actions and saves it as:

        training_records/ep{joint_action_id}.csv
    """

    final_records_folder = Path(final_records_folder).resolve()
    final_records_folder.mkdir(parents=True, exist_ok=True)

    tmp_records_folder = (final_records_folder / f"_tmp_joint_action_{joint_action_id}").resolve()

    if tmp_records_folder.exists():
        shutil.rmtree(tmp_records_folder)

    tmp_records_folder.mkdir(parents=True, exist_ok=True)

    tmp_paths_csv = prepare_tmp_records_folder_for_routerl(
        tmp_records_folder=tmp_records_folder,
        final_records_folder=final_records_folder,
    )

    av_actions, human13_action = joint_id_to_avs_and_human13(
        joint_action_id,
        nb_avs=10,
    )

    if count_human13_in_trafficlight:
        trafficlight_actions = av_actions + [human13_action]
    else:
        trafficlight_actions = av_actions

    tl_0, tl_1, tl_y_used = trafficlight_from_joint_action(trafficlight_actions)

    print("\n==================================================")
    print("Running joint_action_id:", joint_action_id)
    print("AV actions:", av_actions)
    print("Human 13 action:", human13_action)
    print("Number on route 1 used for traffic light:", sum(trafficlight_actions))
    print("Traffic lights:", tl_0, tl_1, tl_y_used)
    print("Temporary folder:", tmp_records_folder)
    print("Temporary paths.csv:", tmp_paths_csv)
    print("Output file:", final_records_folder / "episodes" / f"ep{joint_action_id}.csv")
    print("==================================================\n")

    if not tmp_paths_csv.exists():
        raise FileNotFoundError(
            f"paths.csv disappeared before simulation started:\n{tmp_paths_csv}"
        )

    update_trafficlight(tl_0, tl_1, tl_y_used, offset=offset)

    success = False
    output_csv = None

    try:
        simulate_avs_plus_human13(
            records_folder=str(tmp_records_folder),  # absolute path
            av_actions=av_actions,
            human13_action=human13_action,
            nb_agents=15,
            open_gui=False,
            human13_id=human13_id,
        )

        output_csv = move_single_episode_csv(
            tmp_records_folder=tmp_records_folder,
            final_records_folder=final_records_folder,
            episode_id=joint_action_id,
        )

        success = True

    finally:
        if success:
            if tmp_records_folder.exists():
                shutil.rmtree(tmp_records_folder)
        else:
            print("\nSimulation failed.")
            print("Keeping temporary folder for debugging:")
            print(tmp_records_folder)
            print("Check whether this file exists:")
            print(tmp_paths_csv)

    return {
        "joint_action_id": joint_action_id,
        "episode_file": str(output_csv),
        "av_actions": "".join(map(str, av_actions)),
        "human13_action": human13_action,
        "num_avs_route_1": sum(av_actions),
        "num_route_1_including_human13": sum(av_actions) + human13_action,
        "trafficlight_count_includes_human13": count_human13_in_trafficlight,
        "tl_0": tl_0,
        "tl_1": tl_1,
        "tl_y": tl_y_used,
    }

def run_all_joint_actions_with_human13_single_folder(
    final_records_folder="training_records",
    offset=0,
    human13_id=13,
    count_human13_in_trafficlight=False,
    start_joint_action_id=0,
    end_joint_action_id=2048,
):
    """
    Runs all 2048 joint actions.

    Output:
        training_records/ep0.csv
        training_records/ep1.csv
        ...
        training_records/ep2047.csv

    Also writes:
        training_records/joint_action_manifest.csv
    """

    final_records_folder = Path(final_records_folder)
    final_records_folder.mkdir(parents=True, exist_ok=True)

    manifest_path = final_records_folder / "joint_action_manifest.csv"

    rows = []

    for joint_action_id in range(start_joint_action_id, end_joint_action_id):
        row = run_one_joint_action_with_human13_single_folder(
            joint_action_id=joint_action_id,
            final_records_folder=final_records_folder,
            offset=offset,
            human13_id=human13_id,
            count_human13_in_trafficlight=count_human13_in_trafficlight,
        )

        rows.append(row)

        # Save after every episode, so progress is preserved if SUMO crashes.
        pd.DataFrame(rows).to_csv(manifest_path, index=False)

    return pd.DataFrame(rows)



def run_joint_action_ids(ids, offset=0):
    times = get_next_training_record_index("training_records")

    for id in ids:
        s = id_to_strategy(id)
        run_one_joint_action(s, num_times=times, offset=offset)
        times += 1

# ============================================================
# Simulate all joint actions of:
#   - all 10 AVs
#   - human agent with id == 13
#
# All other humans are fixed to route 0.
# ============================================================

def int_to_strategy(idx, width):
    """
    Converts integer idx into a binary strategy list of length width.

    Example:
        int_to_strategy(0, 4)  -> [0, 0, 0, 0]
        int_to_strategy(1, 4)  -> [0, 0, 0, 1]
        int_to_strategy(15, 4) -> [1, 1, 1, 1]
    """
    if idx < 0 or idx >= 2**width:
        raise ValueError(f"idx must be between 0 and {2**width - 1}")

    return [int(x) for x in format(idx, f"0{width}b")]


def joint_id_to_avs_and_human13(joint_action_id, nb_avs=10):
    """
    Joint-action encoding:

        bits[0:10] -> AV actions
        bits[10]   -> human 13 action

    Example with nb_avs=10:
        joint_action_id = 0
            AVs      = [0,0,0,0,0,0,0,0,0,0]
            human 13 = 0

        joint_action_id = 1
            AVs      = [0,0,0,0,0,0,0,0,0,0]
            human 13 = 1

        joint_action_id = 2
            AVs      = [0,0,0,0,0,0,0,0,0,1]
            human 13 = 0
    """

    bits = int_to_strategy(joint_action_id, nb_avs + 1)

    av_actions = bits[:nb_avs]
    human13_action = bits[nb_avs]

    return av_actions, human13_action


def _agent_sort_key(agent):
    """
    Robust sorting helper for agents whose ids may be int-like or string-like.
    """
    try:
        return (0, int(agent.id))
    except Exception:
        return (1, str(agent.id))


def set_human_actions_only_human13_variable(
    env,
    human13_action,
    human13_id=13,
    fixed_human_action=0,
):
    """
    Sets human 13 to the chosen action.
    Sets all other humans to fixed_human_action, default route 0.
    """

    env.human_agents = sorted(env.human_agents, key=_agent_sort_key)

    found_human13 = False

    for human in env.human_agents:
        if int(human.id) == int(human13_id):
            human.default_action = int(human13_action)
            found_human13 = True
        else:
            human.default_action = int(fixed_human_action)

        print(f"Human {human.id} default action is {human.default_action}")

    if not found_human13:
        available_ids = [human.id for human in env.human_agents]
        raise ValueError(
            f"Could not find human agent with id {human13_id}. "
            f"Available human ids are: {available_ids}"
        )


def run_one_joint_action_with_human13(
    joint_action_id,
    num_times,
    offset=0,
    human13_id=13,
    count_human13_in_trafficlight=False,
):
    """
    Runs one joint action consisting of:
        - 10 AV actions
        - one action for human id 13
    """

    av_actions, human13_action = joint_id_to_avs_and_human13(
        joint_action_id,
        nb_avs=10,
    )

    if count_human13_in_trafficlight:
        trafficlight_actions = av_actions + [human13_action]
    else:
        trafficlight_actions = av_actions

    tl_0, tl_1, tl_y_used = trafficlight_from_joint_action(trafficlight_actions)

    print("\n==================================================")
    print("Running joint_action_id:", joint_action_id)
    print("AV actions:", av_actions)
    print("Human 13 action:", human13_action)
    print("Number on route 1 used for traffic light:", sum(trafficlight_actions))
    print("Traffic lights:", tl_0, tl_1, tl_y_used)
    print("Offset:", offset)
    print("Training record index:", num_times)
    print("==================================================\n")

    update_trafficlight(tl_0, tl_1, tl_y_used, offset=offset)

    return simulate_avs_plus_human13(
        num_times=num_times,
        av_actions=av_actions,
        human13_action=human13_action,
        nb_agents=15,
        open_gui=False,
        human13_id=human13_id,
    )


def run_all_joint_actions_with_human13(
    offset=0,
    human13_id=13,
    count_human13_in_trafficlight=False,
    manifest_path="joint_action_manifest_human13.csv",
):
    """
    Runs all 2^11 joint actions:
        - 10 AVs
        - human 13

    Other humans are fixed to route 0.

    Saves a manifest CSV mapping:
        joint_action_id -> training_records index -> AV actions -> human 13 action
    """

    times = get_next_training_record_index("training_records")

    rows = []

    total_joint_actions = 2 ** 11

    for joint_action_id in range(total_joint_actions):
        av_actions, human13_action = joint_id_to_avs_and_human13(
            joint_action_id,
            nb_avs=10,
        )

        run_one_joint_action_with_human13(
            joint_action_id=joint_action_id,
            num_times=times,
            offset=offset,
            human13_id=human13_id,
            count_human13_in_trafficlight=count_human13_in_trafficlight,
        )

        rows.append(
            {
                "joint_action_id": joint_action_id,
                "training_record_index": times,
                "av_actions": "".join(map(str, av_actions)),
                "human13_action": human13_action,
                "num_avs_route_1": sum(av_actions),
                "num_route_1_including_human13": sum(av_actions) + human13_action,
            }
        )

        # Save after every run, so you do not lose the mapping if SUMO crashes midway.
        pd.DataFrame(rows).to_csv(manifest_path, index=False)

        times += 1

    return pd.DataFrame(rows)


manifest = run_all_joint_actions_with_human13_single_folder(
    final_records_folder="training_records",
    offset=0,
    human13_id=13,
    count_human13_in_trafficlight=False,
)