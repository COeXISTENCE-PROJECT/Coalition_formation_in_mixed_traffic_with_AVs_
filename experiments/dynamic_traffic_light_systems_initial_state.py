# copied from all_possible_paths.py
 
### In this script I can try all the different combinations of actions for the AV agents.
import itertools
import os
import pandas as pd
import numpy as np

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

viridis = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=5, vmax=10)


def run(tl_0, tl_1, tl_y, nbagents,open_gui=False):

    '''
    - generates a two-route-trafficlight network with the specified traffic light times (writes a .net.xml file **both** within the RouteRL module (for execution), and within the ../networks/two_route_trafficlight folder (for safekeeping). Make sure both paths exist on the machine otherwise it won't run.);
    - initializes the SUMO environment with the specified number of agents;
    - runs the simulation for all 1024 possible AV joint actions;
    - writes the payoff matrix inside the `reward_df_(tl_0)_(tl_1)_(tl_y)_(nbagents)agents.csv` file.

    Set open_gui to True to open the simulation GUI.
    '''

    filename = "reward_df_%s_%s_%s_%sagents.csv"%(tl_0,tl_1, tl_y, nbagents)
    update_trafficlight(tl_0, tl_1, tl_y)
    simulate(nbagents,open_gui)
    record_experiment(filename)


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



def create_environment(nb_agents=23,open_gui=False):
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
            }
        )
    return env



def simulate(nb_agents=23,open_gui=False):

    env = create_environment(nb_agents,open_gui)

    for human in env.human_agents:
        human.default_action = 0

    env.start()
    env.mutation()

    actions = [0, 1]
    print("env.human_agents", env.human_agents)
    print("env.machine_agents", env.machine_agents)
    print("\n")

    env.reset()

    i = 0
    k = 1
    for combination in itertools.product(actions, repeat=len(env.possible_agents)):
        i += 1
        for action in combination:
            env.step(action)
        if i > k*1024/10:
            print("%s combinations out of 1024 tested, %s%s remaining"%(i,(10-k)*10,"%"))
            k += 1

        env.reset()
    
    env.stop_simulation()


def build_df(i):

    '''
    Returns the payoff matrix from the last experiment as a pandas DataFrame.
    '''
    
    df = pd.read_csv("training_records/episodes/ep"+str(i)+".csv")
    df = df[df["kind"] == "AV"]
    df = df.sort_values(by="start_time").reset_index(drop=True)
    df["reward"] = -1*df["travel_time"]
    df = df[["reward","action"]]
    return df

def write_line(i,df):
    '''
    Auxiliary function for `record_experiment()`
    '''
    line = str(i)
    for i in range(10):
        line = line + "," + str(df["reward"].values[i])
    return line

def record_experiment(file_name):

    '''
    Writes down the payoff matrix from the last experiment into the file (file name entered as input)
    '''

    with open(file_name, "w") as f:
        f.write("id,0,1,2,3,4,5,6,7,8,9\n")
        for i in range(1024):
            data = build_df(i+1)
            text = write_line(i,data)
            f.write(text+"\n")

def run_exp(tl_0, tl_1, tl_y, nbagents,nbsamples=1,open_gui=False):

    '''
    Calculates expected values of the payoff matrix for the specified two-route-trafficlight network, over 'nbsamples' runs made with offset values uniformly distributed over the total cycle length.
    '''
    cyclelength = tl_0 + tl_1 + 2*tl_y
    les_offsets = [i*cyclelength/nbsamples for i in range(nbsamples)]
    tab = np.zeros((10,1024))
    
    for i in range(nbsamples):
        offset = les_offsets[i]
        filename = "reward_df_%s_%s_%s_%sagents_%soffset.csv"%(tl_0,tl_1, tl_y, nbagents,offset)
        try:
            tab_reward = pd.read_csv(filename)
        except:
            print("Reward table for tl_0 = %s, tl_1 = %s, tl_y = %s and %s agents, with offset = %s, was not found. Running..."%(tl_0,tl_1,tl_y,nbagents,offset))
            update_trafficlight(tl_0, tl_1, tl_y, offset)
            simulate(nbagents,open_gui)
            record_experiment(filename)
            tab_reward = pd.read_csv("reward_df_%s_%s_%s_%sagents.csv"%(tl_0,tl_1,tl_y,nbagents))
        for i in range(10):
            tab[i] += tab_reward[i]
            #tab[i] += tab_reward[str(i)].values
    
    tab = tab/nbsamples
    filename = "reward_df_%s_%s_%s_%sagents_%ssamples.csv"%(tl_0,tl_1, tl_y, nbagents,nbsamples)
    with open(filename, "w") as f:
        f.write("id,0,1,2,3,4,5,6,7,8,9\n")
        for row in range(1024):
            line = str(row)
            for i in range(10):
                line = line + "," + str(tab[i][row])
            f.write(line+"\n")
    

def simulate_action(strategy,nb_agents=23,open_gui=True):

    action = (x for x in strategy)

    env = create_environment(nb_agents,open_gui)
    env.start()
    env.mutation()

    print("env.human_agents", env.human_agents)
    print("env.machine_agents", env.machine_agents)
    print("\n")

    env.reset()
    env.step(action)
    
    env.stop_simulation()


def simulate_coalition(coalition,nb_agents=23):

    action = coalition_to_strategy(coalition)
    simulate_action(action,nb_agents)


def id_to_strategy(id):
    '''
    INPUT :`id`= integer from`[0,1023]`

    OUTPUT :`strategy`= array of size 10, binary representation of number`id`
    '''
    strategy = [0 for _ in range(10)]
    i = 0
    while id > 0:
        if id % 2 == 1:
            strategy[9-i] = 1
        id = id//2
        i += 1
    return strategy


def strategy_to_id(s):
    '''
    INPUT :`strategy`= array of size 10, binary representation of number`id`

    OUTPUT :`id`= integer within`[0,1023]`
    '''
    id = 0
    for i in range(10):
        id += s[i]*(2**(9-i))
    return id


def coalition_to_strategy(coalition):
    '''
    INPUT :`coalition`= subset of [0,9]

    OUTPUT :`strategy`= array of size 10,`strategy[i] = 1`iff i is in the coalition

    NOTE :`strategy_to_coalition(id_to_strategy())`may be used to map an integer to a subset of [0,9]
    '''
    s = [0 for _ in range(10)]
    for i in coalition:
        s[i] = 1
    return s


def strategy_to_coalition(s):
    '''
    INPUT :`strategy`= array of size 10
    
    OUTPUT :`coalition`= subset of [0,9],`strategy[i] = 1`iff i is in the coalition

    NOTE :`strategy_to_coalition(id_to_strategy())`may be used to map an integer to a subset of [0,9]
    '''
    coalition = []
    for i in range(10):
        if s[i] == 1:
            coalition.append(i)
    return coalition


def neighbouring_strategies(s):

    '''
    INPUT :`s`= array of size n, describing the joint actions of the 10 AVs (route 0 or 1) \\
    OUTPUT : array of all 10 joint actions, each one with one of the AVs deviating from`s`
    '''

    neigh = [ [s[j] for j in range(10)] for _ in range(10)]
    for i in range(10):
        neigh[i][i] = 1 - neigh[i][i]
    return neigh


def neighbouring_ids(id):
    
    '''
    goes from the space of ids (integers in`0..1023`) to strategies, then to neighbouring strategies (see`neighbouring_strategies()`), and back to ids.
    '''

    s = id_to_strategy(id)
    t = neighbouring_strategies(s)
    neigh = []
    for i in range(10):
        neigh.append(strategy_to_id(t[i]))
    return neigh


def build_rewardtable(file_name):

    '''
    Returns the payoff matrix from an experiment (stored in the file given as input) as an array.

    !! Will crash if the file does not exist.
    '''

    df_reward = pd.read_csv(file_name)
    tab = np.zeros((10,1024))
    for i in range(10):
        tab[i] = df_reward[str(i)].values
    return tab


def custom_df(tl_list, custom_filename = "reward_df_custom.csv"):

    '''
    `tl_list`is an array of size 11. For each i in [0...10],`tl_list[i]`is a tuple`(tl_0,tl_1,tl_y,nb_agents)`, representing specific parameters of a network with static traffic lights.

    `custom_df`builds the payoff matrix of a network with dynamic traffic lights, where whenever n AVs choose route 1, traffic lights are set to`tl_list[n]`
    '''

    assert len(tl_list) == 11

    f = open(custom_filename,"w")
    
    f.write("id,0,1,2,3,4,5,6,7,8,9\n")
    for id in range(1024):

        s = id_to_strategy(id)
        n_1 = 0
        for j in range(10):
            n_1 += s[j]
        tl_0,tl_1,tl_y,nb_agents = tl_list[n_1]

        try:
            df_reward = pd.read_csv("reward_df_%s_%s_%s_%sagents.csv"%(tl_0,tl_1,tl_y,nb_agents))
        except:
            print("Reward table for tl_0 = %s, tl_1 = %s, tl_y = %s and %s agents was not found. Running..."%(tl_0,tl_1,tl_y,nb_agents))
            run(tl_0,tl_1,tl_y,nb_agents)
            df_reward = pd.read_csv("reward_df_%s_%s_%s_%sagents.csv"%(tl_0,tl_1,tl_y,nb_agents))

        text = str(id)
        tab_reward = [[] for _ in range(10)]
        for i in range(10):
            tab_reward[i] = df_reward[str(i)].values
            text = text + "," + str(float(tab_reward[i][id]))

        f.write(text+"\n")

def reward(i,id):
    return float(tab_reward[i][id])


def id_to_reward(id):
    return [reward(i,id) for i in range(10)]


def s_to_reward(s):
    id = strategy_to_id(s)
    return id_to_reward(id)


def nash_deviation(s):
    
    '''
    for a given strategy/joint action`s`, returns an array of size 10`dev`.

    For i in 0..9,`dev[i]`is what the AV number i can gain by deviating ALONE from`s`.
    '''

    rew = s_to_reward(s)
    neigh = neighbouring_strategies(s)
    dev = []
    for i in range(10):
        alternative = s_to_reward(neigh[i])[i]
        dev.append(alternative - rew[i])
    return dev



def nash_equilibrium(s):
    
    '''
    runs`nash_deviation(s)`, verifies if any AV can gain by deviating from`s`, and answers whether`s`is a Nash equilibrium.
    '''

    dev = nash_deviation(s)
    for i in range(10):
        if dev[i] > 0:
            return False
    return True



def strong_nash_equilibrium(s,verbose=True):
    
    '''
    verifies that no subset (aka coalition) of the set of AVs can deviate from`s`. \\
    A coalition deviates if every member increases its payoff when the whole coalition deviates.\\
    Add `verbose=False` if you do not want to print coalitions eligible for a deviation.
    '''

    rew = s_to_reward(s)
    list_c = []
    for id in range(1,1024):
        coalition = strategy_to_coalition(id_to_strategy(id))
        neigh_s = [s[i] for i in range(10)]
        for i in coalition:
            neigh_s[i] = 1 - s[i]
        alt = s_to_reward(neigh_s)
        coalition_deviates = True
        for i in coalition:
            if alt[i] - rew[i] <= 0:
                coalition_deviates = False
        if coalition_deviates:
            list_c.append(coalition)
    if verbose:
        for c in list_c:
            print(c)
    return len(list_c) == 0


def individual_deviations(s):
    
    '''
    runs`nash_deviation(s)`, and returns the list of AVs that can gain by deviating from`s`.
    '''

    dev = nash_deviation(s)
    list_i = []
    for i in range(10):
        # we do not verify if the first AV can deviate. Otherwise, no non-trivial NE will be found.
        if dev[i] > 0:
            list_i.append(i)
    return list_i


def coalition_deviations(s):
    
    '''
    Returns coalitions that may deviate from `s`.
    '''

    rew = s_to_reward(s)
    list_c = []
    for id in range(1,1024):
        coalition = strategy_to_coalition(id_to_strategy(id))
        neigh_s = [s[i] for i in range(10)]
        for i in coalition:
            neigh_s[i] = 1 - s[i]
        alt = s_to_reward(neigh_s)
        coalition_deviates = True
        for i in coalition:
            if alt[i] - rew[i] <= 0:
                coalition_deviates = False
        if coalition_deviates:
            list_c.append(coalition)
    return list_c

# Checking under which parameters of a dynamic-traffic-light network, having all AVs on route 0 is a (strong) Nash equilibrium.
# Said parameters are the traffic light durations with few AVs on route 1, and durations with more AVs on route 1.
# 'threshold' is the number of AVs on route 1 necessary to switch between these two modes.

snash_x, snash_y = [], []
nash_x, nash_y = [], []
notnash_x, notnash_y = [], []

nb_agents = 15      # number of human + AV agents in the network
threshold = 2       # number of AVs on route 1 beyond which the traffic lights adapt
tl_y = 5            # length of each yellow light phase
cycle_length = 50   # total length of the traffic light cycle

for tl_0_below_threshold in range(15,30):
    for tl_0_above_threshold in range(1,tl_0_below_threshold):

        # build traffic light system and payoff matrix
        tl_1_below_threshold = cycle_length - 2*tl_y - tl_0_below_threshold
        tl_1_above_threshold = cycle_length - 2*tl_y - tl_0_above_threshold
        print("Green light durations on each route are (%s,%s) below threshold, (%s,%s) above threshold"%(tl_0_below_threshold,tl_1_below_threshold,tl_0_above_threshold,tl_1_above_threshold))
        tl_list = [(tl_0_below_threshold,tl_1_below_threshold,tl_y,nb_agents) if nb_1 < threshold else (tl_0_above_threshold,tl_1_above_threshold,tl_y,nb_agents) for nb_1 in range(11)]
        custom_df(tl_list)
        tab_reward = build_rewardtable("reward_df_custom.csv")

        # check equilibrium status
        if nash_equilibrium(id_to_strategy(0)):
            if strong_nash_equilibrium(id_to_strategy(0)):
                snash_x.append(tl_0_below_threshold)
                snash_y.append(tl_0_above_threshold)
            else:
                nash_x.append(tl_0_below_threshold)
                nash_y.append(tl_0_above_threshold)
        else:
            notnash_x.append(tl_0_below_threshold)
            notnash_y.append(tl_0_above_threshold)

"""plt.scatter(snash_x, snash_y, color="g", label="Nash and strong equilibria")
plt.scatter(nash_x, nash_y, color="y", label="Nash but not strong equilibria")
plt.scatter(notnash_x, notnash_y, color="r", label="not Nash equilibria")
plt.xlabel("Green light duration on route 0, when n1 < %s"%(threshold))
plt.ylabel("Green light duration on route 0, when n1 >= %s"%(threshold))
plt.legend()
plt.title("Is all-AVs-on-route-0 a (strong) Nash equilibrium, with dynamic traffic light times? Nbagents = %s, threshold = %s"%(nb_agents,threshold))
plt.show()"""

from matplotlib.lines import Line2D

plt.style.use("default")
tnrfont = {'fontname': 'Times New Roman'}

fig, ax = plt.subplots()
fig.set_size_inches(6, 5)

# Background and axis style
ax.set_facecolor("aliceblue")
ax.set_box_aspect(1)
ax.grid(axis="both", which="major", zorder=0, alpha=0.35)
ax.set_axisbelow(True)

# Scatter plots
ax.scatter(
    snash_x, snash_y,
    marker="o",
    s=55,
    facecolor="g",
    edgecolor="0",
    linewidth=0.8,
    alpha=0.85,
    zorder=3,
    label=r"State $x^0$ is a SE")

ax.scatter(
    nash_x, nash_y,
    marker="o",
    s=55,
    facecolor="#FFD700",
    edgecolor="0",
    linewidth=0.8,
    alpha=0.85,
    zorder=3,
    label=r"State $x^0$ is a NE")


ax.scatter(
    notnash_x, notnash_y,
    marker="o",
    s=55,
    facecolor="r",
    edgecolor="0",
    linewidth=0.8,
    alpha=0.85,
    zorder=3,
    label=r"State $x^0$ is not a NE")

# Labels
ax.set_xlabel(
    r"Green time on Route 0 when less than %s AVs choose Route 1" % threshold,
    **tnrfont,
    size=12
)

ax.set_ylabel(
    r"Green time on Route 0 when at least %s AVs choose Route 1" % threshold,
    **tnrfont,
    size=12
)

# Ticks
ax.set_xticks(range(15, 30, 2))
ax.set_yticks(range(0, 30, 2))

ax.set_xticklabels(ax.get_xticks(), **tnrfont, size=12)
ax.set_yticklabels(ax.get_yticks(), **tnrfont, size=12)

# Title
ax.set_title(
    r"Stability of state $x^0$ under different dynamic traffic light durations",

    **tnrfont,
    size=12
)

ax.tick_params(axis="both", labelsize=10)

#"Is all-AVs-on-route-0 a (strong) Nash equilibrium?\n"
#     "Dynamic traffic light times, Nagents = %s, threshold = %s"
#   % (nb_agents, threshold),


# Legend styled similarly to your other figure
legend = ax.legend(
    loc="upper left",
    prop={"family": "Times New Roman", "size": 14},
    frameon=True
)
legend.get_frame().set_edgecolor("0")
legend.get_frame().set_linewidth(0.8)

plt.tight_layout()

plt.savefig("../imgs/dynamic_traffic_light_equilibria.png", dpi=300)
plt.savefig("../imgs/dynamic_traffic_light_equilibria.svg")

plt.show()