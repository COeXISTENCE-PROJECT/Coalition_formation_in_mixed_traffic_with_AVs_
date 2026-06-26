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

threshold = 2
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




def create_environment(num_times, nb_agents=23, open_gui=False):
    sumo_type = "sumo-gui" if open_gui else "sumo" # opening the GUI might be useful for debugging.
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
            "records_folder": f"training_records{num_times}",
            }
        )
    return env

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

def simulate(num_times, nb_agents=23, open_gui=False):

    env = create_environment(num_times, nb_agents,  open_gui)
    env.start()
    env.mutation()

    actions = [0, 1]
    print("env.human_agents", env.human_agents)
    print("env.machine_agents", env.machine_agents)
    print("\n")

    env.reset()

    combination = [1, 1, 0, 0, 1, 1, 0, 0, 0, 0]
    human_rng = random.Random()
    human_rng.seed(None)

    human_combination = num_times_to_human_strategy(num_times, nb_humans=5)

    print("Human combination:", human_combination)

    env.human_agents = sorted(env.human_agents, key=lambda h: h.id)

    for human, action in zip(env.human_agents, human_combination):
        human.default_action = action
        print(f"Human {human.id} default action is {action}")
    
    for action in combination:
        env.step(action)
    
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

def run_one_joint_action(s, num_times, offset=0):
    """
    s is a list of 10 AV actions.
    s[i] = 0 means AV i takes route 0
    s[i] = 1 means AV i takes route 1
    """

    tl_0, tl_1, tl_y_used = trafficlight_from_joint_action(s)

    print("Running joint action:", s)
    print("Number of AVs on route 1:", sum(s))
    print("Traffic lights:", tl_0, tl_1, tl_y_used)
    print("Offset:", offset)

    update_trafficlight(tl_0, tl_1, tl_y_used, offset=offset)

    # IMPORTANT:
    # Replace this with your existing function/code that runs SUMO
    # for one fixed joint action.
    #
    # For example, if your script has something like:
    # run_simulation(s)
    #
    # then call:
    # return run_simulation(s)

    return simulate(num_times, nb_agents=15, open_gui=True)




def run_joint_action_ids(ids, offset=0):
    times = get_next_training_record_index("training_records")

    for id in ids:
        s = id_to_strategy(id)
        run_one_joint_action(s, num_times=times, offset=offset)
        times += 1

# ============================================================
# Example: put here the joint-action ids that caused deviation
# ============================================================

"""joint_action_ids = [
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,
    792,

]"""
joint_action_ids = [
    816,816,816,816,816,816,816,816,816,816,816,816,816,816,816,816,816,816,816,
    816,816,816,816,816,816,816,816,816,816,816,816, 816]

print("len of joint action ids is: ", len(joint_action_ids), "\n\n")

run_joint_action_ids(joint_action_ids, offset=0)