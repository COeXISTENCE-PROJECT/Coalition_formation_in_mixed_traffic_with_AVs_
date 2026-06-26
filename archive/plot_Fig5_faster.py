# copied from all_possible_paths.py
 
### In this script I can try all the different combinations of actions for the AV agents.
import itertools
import os
import pandas as pd
import numpy as np

#from routerl import TrafficEnvironment


#from routerl.keychain import Keychain as kc
#from routerl.utilities import get_params

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

mpl.rcParams["font.family"] = "Times New Roman"

# Make math text visually compatible with Times New Roman
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["mathtext.rm"] = "STIXGeneral"
mpl.rcParams["mathtext.it"] = "STIXGeneral:italic"
mpl.rcParams["mathtext.bf"] = "STIXGeneral:bold"

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


from matplotlib.lines import Line2D

plt.style.use("default")
tnrfont = {'fontname': 'Times New Roman'}

# ============================================================
# Compute stability regions for a given total number of agents
# ============================================================

from functools import lru_cache

# Precompute useful ids once
ALL_IDS = np.arange(1024)
NB_ROUTE_1 = np.array([bin(i).count("1") for i in ALL_IDS])

COALITION_IDS = np.arange(1, 1024)
COALITION_MASKS = np.array(
    [id_to_strategy(i) for i in COALITION_IDS],
    dtype=bool
)

SINGLE_DEVIATION_IDS = np.array(
    [strategy_to_id(coalition_to_strategy([i])) for i in range(10)]
)


@lru_cache(maxsize=None)
def cached_reward_table(tl_0, tl_1, tl_y, nb_agents):
    filename = f"reward_df_{tl_0}_{tl_1}_{tl_y}_{nb_agents}agents.csv"

    if not os.path.exists(filename):
        print(
            f"Reward table {filename} was not found. Running simulation..."
        )
        run(tl_0, tl_1, tl_y, nb_agents)

    return build_rewardtable(filename)


def build_dynamic_reward_table_fast(tl_list):
    """
    Fast in-memory replacement for custom_df(tl_list) + build_rewardtable("reward_df_custom.csv").
    """
    tab = np.empty((10, 1024))

    for nb_1 in range(11):
        tl_0, tl_1, tl_y, nb_agents = tl_list[nb_1]
        static_tab = cached_reward_table(tl_0, tl_1, tl_y, nb_agents)

        cols = NB_ROUTE_1 == nb_1
        tab[:, cols] = static_tab[:, cols]

    return tab


def classify_initial_state_fast(tab_reward):
    """
    Classifies x0 = everyone on route 0 as:
    - "snash"    : no one wants to deviate, strong Nash
    - "nash"     : Nash but not strong, clubs are formed
    - "notnash"  : some agents want to deviate
    """
    rew = tab_reward[:, 0]

    # Individual deviations from x0
    individual_gain = tab_reward[np.arange(10), SINGLE_DEVIATION_IDS] - rew

    if np.any(individual_gain > 0):
        return "notnash"

    # Coalition deviations from x0
    alt = tab_reward[:, COALITION_IDS].T
    gain = alt - rew

    coalition_improves = gain > 0

    # For each coalition, only check members of that coalition
    coalition_deviates = np.all(
        np.where(COALITION_MASKS, coalition_improves, True),
        axis=1
    )

    if np.any(coalition_deviates):
        return "nash"

    return "snash"

def compute_stability_points(nb_agents, threshold=2, tl_y=5, cycle_length=50):
    snash_x, snash_y = [], []
    nash_x, nash_y = [], []
    notnash_x, notnash_y = [], []

    for tl_0_below_threshold in range(15, 27):
        for tl_0_above_threshold in range(1, 26):

            tl_1_below_threshold = cycle_length - 2 * tl_y - tl_0_below_threshold
            tl_1_above_threshold = cycle_length - 2 * tl_y - tl_0_above_threshold

            tl_list = [
                (
                    tl_0_below_threshold,
                    tl_1_below_threshold,
                    tl_y,
                    nb_agents
                )
                if nb_1 < threshold
                else
                (
                    tl_0_above_threshold,
                    tl_1_above_threshold,
                    tl_y,
                    nb_agents
                )
                for nb_1 in range(11)
            ]

            tab_reward_fast = build_dynamic_reward_table_fast(tl_list)
            status = classify_initial_state_fast(tab_reward_fast)

            if status == "snash":
                snash_x.append(tl_0_below_threshold)
                snash_y.append(tl_0_above_threshold)
            elif status == "nash":
                nash_x.append(tl_0_below_threshold)
                nash_y.append(tl_0_above_threshold)
            else:
                notnash_x.append(tl_0_below_threshold)
                notnash_y.append(tl_0_above_threshold)

    return {
        "snash_x": snash_x,
        "snash_y": snash_y,
        "nash_x": nash_x,
        "nash_y": nash_y,
        "notnash_x": notnash_x,
        "notnash_y": notnash_y,
    }


# ============================================================
# Plot one panel
# ============================================================

def plot_stability_panel(
    ax,
    data,
    title,
    show_ylabel=False,
    show_xlabel=True,
    show_legend=False,
    xlim=None,
    ylim=None,
    xticks=None,
    yticks=None
):
    # Requested palette
    col_no_dev = "steelblue"      # grey-blue
    col_dev    = "darkseagreen"   # grey-green
    col_club   = "firebrick"      # clubs are formed

    ax.set_facecolor("white")
    ax.set_box_aspect(1.0)
    ax.grid(axis="both", which="major", zorder=0, alpha=0.35)
    ax.set_axisbelow(True)

    ax.scatter(
        data["snash_x"],
        data["snash_y"],
        marker="o",
        s=250,
        facecolor=col_no_dev,
        edgecolor="black",
        linewidth=0.9,
        alpha=0.90,
        zorder=3,
        label=r"No one wants to deviate"
    )

    ax.scatter(
        data["nash_x"],
        data["nash_y"],
        marker="o",
        s=400,
        facecolor=col_club,
        edgecolor="black",
        linewidth=1.1,
        alpha=0.95,
        zorder=5,
        label=r"Clubs are formed"
    )

    ax.scatter(
        data["notnash_x"],
        data["notnash_y"],
        marker="o",
        s=250,
        facecolor=col_dev,
        edgecolor="black",
        linewidth=0.9,
        alpha=0.90,
        zorder=3,
        label=r"Some AVs want to deviate"
    )

    ax.set_title(title, **tnrfont, math_fontfamily="stix", size=32)

    if show_xlabel:
        ax.set_xlabel(
            r"$G_1$",
            **tnrfont,
            size=28
        )
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel(
            #"Actuated green time on Route 0\n after traffic light adaptation",
            r"$G'_1$",
            **tnrfont,
            size=28
        )
    else:
        ax.set_ylabel("")

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xticks is not None:
        ax.set_xticks(xticks)
    else:
        ax.set_xticks(range(15, 28, 2))

    if yticks is not None:
        ax.set_yticks(yticks)
    else:
        ax.set_yticks(range(0, 28, 2))

    ax.set_xticklabels(ax.get_xticks(), **tnrfont, size=21)
    ax.set_yticklabels(ax.get_yticks(), **tnrfont, size=21)

    ax.tick_params(axis="both", labelsize=21, width=1.4, length=4)

    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    if show_legend:
        legend = ax.legend(
            loc="upper left",
            prop={"family": "Times New Roman", "size": 18},
            frameon=True
        )
        legend.get_frame().set_edgecolor("0")
        legend.get_frame().set_linewidth(0.8)


# ============================================================
# Compute both cases
# ============================================================
"""
threshold = 2
tl_y = 5
cycle_length = 50

data_15 = compute_stability_points(
    nb_agents=15,
    threshold=3,
    tl_y=tl_y,
    cycle_length=cycle_length
)

data_23 = compute_stability_points(
    nb_agents=23,
    threshold=3,
    tl_y=tl_y,
    cycle_length=cycle_length
)


# ============================================================
# Two-panel figure
# ============================================================

# ============================================================
# Two-panel figure
# ============================================================

fig, axes = plt.subplots(1, 2, sharey=False)
fig.set_size_inches(12.2, 9.5)

plot_stability_panel(
    axes[0],
    data_15,
    title=r"a) 15 agents",
    show_ylabel=True,
    show_legend=False,
    xlim=(14.5, 26.5),
    ylim=(0.5, 25.5),
    xticks=range(15, 29, 2),
    yticks=range(0, 28, 2)
)

plot_stability_panel(
    axes[1],
    data_23,
    title=r"b) 23 agents",
    show_ylabel=False,
    show_legend=False,
    xlim=(14.5, 26.5),
    ylim=(0.5, 25.5),
    xticks=range(15, 27, 2),
    yticks=range(0, 28, 2)
)

# Shared legend on top
handles, labels = axes[0].get_legend_handles_labels()

legend = fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    prop={"family": "Times New Roman", "size": 24},
    handlelength=1.4,
    handletextpad=0.2,
    columnspacing=1.4,
)

fig.subplots_adjust(
    left=0.075,
    right=0.985,
    bottom=0.12,
    top=0.86,
    wspace=0.12,
)

plt.savefig("../imgs/dynamic_traffic_light_equilibria_15_23.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
plt.savefig("../imgs/dynamic_traffic_light_equilibria_15_23.svg", bbox_inches="tight", pad_inches=0.03)

plt.show()"""


# ============================================================
# Compute all four cases
# ============================================================

tl_y = 5
cycle_length = 50

# Top row: threshold = 2
data_15_t2 = compute_stability_points(
    nb_agents=15,
    threshold=2,
    tl_y=tl_y,
    cycle_length=cycle_length
)

data_23_t2 = compute_stability_points(
    nb_agents=23,
    threshold=2,
    tl_y=tl_y,
    cycle_length=cycle_length
)

# Bottom row: threshold = 3
data_15_t3 = compute_stability_points(
    nb_agents=15,
    threshold=3,
    tl_y=tl_y,
    cycle_length=cycle_length
)

data_23_t3 = compute_stability_points(
    nb_agents=23,
    threshold=3,
    tl_y=tl_y,
    cycle_length=cycle_length
)

def count_club_dots(data):
    return len(data["nash_x"])

print("Number of red dots / club-formed cases:")
print("a) 15 agents, q1_thr = 2:", count_club_dots(data_15_t2))
print("b) 23 agents, q1_thr = 2:", count_club_dots(data_23_t2))
print("c) 15 agents, q1_thr = 3:", count_club_dots(data_15_t3))
print("d) 23 agents, q1_thr = 3:", count_club_dots(data_23_t3))

total_red_dots = (
    count_club_dots(data_15_t2)
    + count_club_dots(data_23_t2)
    + count_club_dots(data_15_t3)
    + count_club_dots(data_23_t3)
)

print("Total red dots across all panels:", total_red_dots)

# ============================================================
# Four-panel figure: rows = thresholds, columns = number of agents
# ============================================================

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, gridspec_kw={"wspace": 0.01, "hspace": 0.20})
fig.set_size_inches(12, 14.5)

common_kwargs = dict(
    show_legend=False,
    xlim=(14.5, 26.5),
    ylim=(0.5, 25.5),
    xticks=range(15, 27, 2),
    yticks=range(0, 28, 2)
)

plot_stability_panel(
    axes[0, 0],
    data_15_t2,
    title=r"a) 15 agents, $q_1^{\mathrm{thr}} = 2$",
    show_ylabel=True,
    show_xlabel=False,
    **common_kwargs
)

plot_stability_panel(
    axes[0, 1],
    data_23_t2,
    title=r"b) 23 agents, $q_1^{\mathrm{thr}} = 2$",
    show_ylabel=False,
    show_xlabel=False,
    **common_kwargs
)

plot_stability_panel(
    axes[1, 0],
    data_15_t3,
    title=r"c) 15 agents, $q_1^{\mathrm{thr}} = 3$",
    show_ylabel=True,
    show_xlabel=True,
    **common_kwargs
)

plot_stability_panel(
    axes[1, 1],
    data_23_t3,
    title=r"d) 23 agents, $q_1^{\mathrm{thr}} = 3$",
    show_ylabel=False,
    show_xlabel=True,
    **common_kwargs
)

handles, labels = axes[0, 0].get_legend_handles_labels()

legend = fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.49, 0.96),
    ncol=3,
    frameon=False,
    prop={"family": "Times New Roman", "size": 23},
    handlelength=1.4,
    handletextpad=0.2,
    columnspacing=1.4,
)

fig.subplots_adjust(
    left=0.075,
    right=0.985,
    bottom=0.12,
    top=0.86,
    wspace=0.02,
    hspace=0.18,
)

plt.savefig("../imgs/dynamic_traffic_light_equilibria_15_23.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
plt.savefig("../imgs/dynamic_traffic_light_equilibria_15_23.svg", bbox_inches="tight", pad_inches=0.03)

plt.show()