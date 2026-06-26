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



"""nb_agents = 15      # number of human + AV agents in the network
threshold = 2       # number of AVs on route 1 beyond which the traffic lights adapt
tl_y = 5            # length of each yellow light phase
cycle_length = 50   # total length of the traffic light cycle

list_t0s = [(20,10)]
for tl_0_below_threshold,tl_0_above_threshold in list_t0s:
    tl_1_below_threshold = cycle_length - 2*tl_y - tl_0_below_threshold
    tl_1_above_threshold = cycle_length - 2*tl_y - tl_0_above_threshold
    print("Green light durations on each route are (%s,%s) below threshold, (%s,%s) above threshold"%(tl_0_below_threshold,tl_1_below_threshold,tl_0_above_threshold,tl_1_above_threshold))

    tl_list = [(tl_0_below_threshold,tl_1_below_threshold,tl_y,nb_agents) if nb_1 < threshold else (tl_0_above_threshold,tl_1_above_threshold,tl_y,nb_agents) for nb_1 in range(11)]
    custom_df(tl_list)
    tab_reward = build_rewardtable("reward_df_custom.csv")

    print("Deviations from x^0 = (0,0,...,0):", nash_deviation(id_to_strategy(0))) # prints the potential gains for each AV to be the first one to reroute from 0 to 1

    # prints all the joint actions that are Nash equilibria
    print("Printing joint actions that are Nash equilibria:")
    for id in range(1024):
        if nash_equilibrium(id_to_strategy(id)):
            print(id_to_strategy(id))
    print("Done")

    print("Is x^0 = (0,0,...,0) a Nash equilibrium? :",nash_equilibrium(id_to_strategy(0)))
    if nash_equilibrium(id_to_strategy(0)):
        print("Is x^0 = (0,0,...,0) a strong equilibrium? :",strong_nash_equilibrium(id_to_strategy(0)))
    print("")"""



# set parameters
"""nb_agents = 23      # number of human + AV agents in the network
threshold = 2       # number of AVs on route 1 beyond which the traffic lights adapt
tl_y = 5            # length of each yellow light phase
cycle_length = 60   # total length of the traffic light cycle

# define dynamic traffic light system
tl_0_below_threshold = 30   # how long the green light lasts on route 0 before traffic lights adapt
tl_0_above_threshold = 25   # how long the green light lasts on route 0 after traffic lights adapt
tl_1_below_threshold = cycle_length - 2*tl_y  - tl_0_below_threshold
tl_1_above_threshold = cycle_length - 2*tl_y  - tl_0_above_threshold"""


# set parameters
nb_agents = 15      # number of human + AV agents in the network
threshold = 3       # number of AVs on route 1 beyond which the traffic lights adapt
tl_y = 5            # length of each yellow light phase
cycle_length = 50   # total length of the traffic light cycle


tl_0_below_threshold = 21   # how long the green light lasts on route 0 before traffic lights adapt
tl_0_above_threshold = 9    # how long the green light lasts on route 0 after traffic lights adapt
# and idem on route 1:
tl_1_below_threshold = cycle_length - 2*tl_y  - tl_0_below_threshold
tl_1_above_threshold = cycle_length - 2*tl_y  - tl_0_above_threshold

# payoff matrix
tl_list = [(tl_0_below_threshold,tl_1_below_threshold,tl_y,nb_agents) if nb_1 < threshold else (tl_0_above_threshold,tl_1_above_threshold,tl_y,nb_agents) for nb_1 in range(11)]
custom_df(tl_list)
tab_reward = build_rewardtable("reward_df_21_19_5_15agents.csv")

# Calculating deviations from the initial situation
s_0 = id_to_strategy(0)
print("#######################\nInitial situation: everyone on route 0.")
print("List of clubs that may form from x^0: \t",coalition_deviations(s_0))
print("Nash equilibrium: %s, strong equilibrium: %s"%(nash_equilibrium(s_0),strong_nash_equilibrium(s_0,verbose=False)))
p_0 = s_to_reward(s_0)
nash_dev = nash_deviation(s_0)

# Preparing the plot
les_x = [i for i in range(10)]
plt.style.use("default")
tnrfont = {'fontname':'Times New Roman'}
# plt.style.use('science')
fig = plt.figure()
fig.set_size_inches(12,3)
list_clubs = [[],[1,5,6],[1,5,6,7],[0,1,5,6,7],[0,1,5,6]]
nb_clubs_tested = len(list_clubs)
les_ax = []

les_xy = []


def plot_deviation(club,p0):

    '''
    Input : `club` is a subset of the set of players.
    Output : `y` is the reward of all players if the club goes to route 1 and all other players go to route 0. \\
            `y_dev` is the reward of each player if they deviate alone from this situation.
    '''

    s = coalition_to_strategy(club)
    print("#######################\nClub is %s.\tJoint action is %s"%(club,s))
    print("List of agents that could deviate individually: \t",individual_deviations(s))
    print("List of coalitions that could deviate simultaneously: \t",coalition_deviations(s))
    print("Nash equilibrium: %s, strong equilibrium: %s"%(nash_equilibrium(s),strong_nash_equilibrium(s,verbose=False)))
    p = s_to_reward(s)
    nash_dev = nash_deviation(s)
    y = [p[i]/p0[i] for i in range(len(p))]
    dev = [nash_dev[i]/p0[i] for i in range(10)]
    return y,dev,s


def plot_clubs(club,shift):
    '''
    Computes payoffs and deviations from the joint action where `club` is on route 1 and everyone else on route 0. \\
    Then, plots the results. Use `shift` to display several results side by side without overlapping.
    '''
    les_y,les_dev,s = plot_deviation(club,p_0)
    for i in range(10):
        color = '0'
        if les_dev[i] < 0:
            color = 'r'
            les_xy.append((les_x[i],les_y[i] + les_dev[i]))
        les_ax[-1].arrow(i+shift,les_y[i],0,les_dev[i],width=0.1,head_width = 0.35, head_length = min(0.1,np.abs(les_dev[i])),length_includes_head = True,color=color,zorder=2,alpha = 0.8)
        if s[i] == 0:
            les_ax[-1].plot(i+shift,les_y[i],"o",markerfacecolor='0',markeredgecolor='0',markersize=5,zorder=3)
        else:
            les_ax[-1].plot(i+shift,les_y[i],"o",markerfacecolor='1',markeredgecolor='0',markersize=5,zorder=3)
        plt.draw()
        # ax.plot([i + shift,i + shift],les_yplusdev[i],color=color,alpha=0.5)


for n in range(nb_clubs_tested):
    club = list_clubs[n]
    les_ax.append(fig.add_subplot(1,nb_clubs_tested,n+1))
    les_ax[-1].set_ylim(0.8,3)
    les_ax[-1].set_xticks([i  for i in range(10)], minor=False, )
    les_ax[-1].set_yticks([0.8,1,1.5,2,2.5,3], minor = False)
    les_ax[-1].set_xticklabels(les_ax[-1].get_xticks(), **tnrfont)
    les_ax[-1].set_yticklabels(les_ax[-1].get_yticks(), **tnrfont)
    les_ax[-1].grid(axis="x",which="major",zorder=0)
    les_ax[-1].axhline(1, color='0', linewidth=1)
    title = "Club is %s"%(club) if n > 0 else "No club"
    les_ax[-1].set_title(title,**tnrfont)
    les_ax[-1].set_box_aspect(1)
    plot_clubs(club,0)

    extent = les_ax[-1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig('../imgs/deviations_%s.png'%(club), bbox_inches=extent.expanded(1.37, 1.37),transparent=True)
    fig.savefig('../imgs/deviations_%s.svg'%(club), bbox_inches=extent.expanded(1.37, 1.37),transparent=True)
    les_ax[-1].yaxis.set_visible(False)

print(les_xy)
con = ConnectionPatch(xyA=les_xy[0], xyB=les_xy[0], connectionstyle="arc3,rad=.1", coordsA="data", coordsB="data",axesA=les_ax[1], axesB=les_ax[4], color="b",arrowstyle="->")
les_ax[4].add_artist(con)
con = ConnectionPatch(xyA=les_xy[1], xyB=les_xy[1], connectionstyle="arc3,rad=-0.1", coordsA="data", coordsB="data",axesA=les_ax[1], axesB=les_ax[2], color="b",arrowstyle="->")
les_ax[4].add_artist(con)
con = ConnectionPatch(xyA=les_xy[2], xyB=les_xy[2], connectionstyle="arc3,rad=.1", coordsA="data", coordsB="data",axesA=les_ax[2], axesB=les_ax[3], color="b",arrowstyle="->")
les_ax[4].add_artist(con)
con = ConnectionPatch(xyA=les_xy[3], xyB=les_xy[3], connectionstyle="arc3,rad=.1", coordsA="data", coordsB="data",axesA=les_ax[3], axesB=les_ax[4], color="b",arrowstyle="->")
les_ax[4].add_artist(con)

#ax1.plot(x[i],y[i],'ro',markersize=10)
#ax2.plot(x[i],y[i],'ro',markersize=10)
les_ax[0].yaxis.set_visible(True)
les_ax[0].set_ylabel("Normalized travel time",**tnrfont)
les_ax[0].set_facecolor('aliceblue')
les_ax[4].set_facecolor('honeydew')

les_ax[0].set_title("a) No club",**tnrfont,size=12)
les_ax[1].set_title("b) Club is [1,5,6]",**tnrfont,size=12)
les_ax[2].set_title("c) Club is [1,5,6,7]",**tnrfont,size=12)
les_ax[3].set_title("d) Club is [0,1,5,6,7]",**tnrfont,size=12)
les_ax[4].set_title("e) Club is [0,1,5,6]",**tnrfont,size=12)

les_ax[2].set_xlabel("Agent ID",**tnrfont,size=12)

black_dot = lines.Line2D([0], [0], color = "w", marker="o",markerfacecolor='0',markeredgecolor='0',markersize=8)
white_dot = lines.Line2D([0], [0], color = "w", marker="o",markerfacecolor='1',markeredgecolor='0',markersize=8)

# leg = plt.legend([black_dot, white_dot], ["Agent on route 0", "Agent on route 1"],loc = 'upper right')

# plt.title("(%s,%s) below threshold, (%s,%s) above threshold, threshold =  %s"%(tl_0_below_threshold,tl_1_below_threshold,tl_0_above_threshold,tl_1_above_threshold,threshold))
plt.savefig("..\imgs\deviations.png")
plt.savefig("..\imgs\deviations.svg")
plt.show()

# define dynamic traffic light system
"""tl_0_below_threshold = 21   # how long the green light lasts on route 0 before traffic lights adapt
tl_0_above_threshold = 9    # how long the green light lasts on route 0 after traffic lights adapt
tl_1_below_threshold = cycle_length - 2*tl_y  - tl_0_below_threshold
tl_1_above_threshold = cycle_length - 2*tl_y  - tl_0_above_threshold


# build payoff matrix
tl_list = [(tl_0_below_threshold,tl_1_below_threshold,tl_y,nb_agents) if nb_1 < threshold else (tl_0_above_threshold,tl_1_above_threshold,tl_y,nb_agents) for nb_1 in range(11)]
custom_df(tl_list)
tab_reward = build_rewardtable("reward_df_21_19_5_15agents_0.0offset_1.csv")

# Calculating deviations from the initial situation
s_0 = id_to_strategy(0)
print("#######################\nInitial situation: everyone on route 0.")
c_dev = coalition_deviations(s_0)
print("List of clubs that may form from x^0:",c_dev)
rew_0 = s_to_reward(s_0)
print("Payoffs:",rew_0)
print("Nash equilibrium: %s, strong equilibrium: %s"%(nash_equilibrium(s_0),strong_nash_equilibrium(s_0,verbose=False)))


def explore_deviations(club,nb_tabs=1):
    s = coalition_to_strategy(club)
    print(nb_tabs*"\t"+"#######################\n"+nb_tabs*"\t"+"Club is %s. Joint action is %s"%(club,s))
    ind_dev = individual_deviations(s)
    print(nb_tabs*"\t"+"List of agents that could deviate individually: %s"%ind_dev)
    new_c_dev = []
    for new_c in coalition_deviations(s):
        outofclub = True
        for i in new_c:
            if i in club:
                outofclub = False
        if outofclub:
            new_c_dev.append(new_c)
    print(nb_tabs*"\t"+"List of coalitions that could deviate simultaneously: %s"%new_c_dev)

    rew = s_to_reward(s)
    print(nb_tabs*"\t"+"Payoffs: %s"%rew)
    print(nb_tabs*"\t"+"Nash equilibrium: %s, strong equilibrium: %s"%(nash_equilibrium(s),strong_nash_equilibrium(s,verbose=False)))
    print("Is x^0 = (0,0,...,0) a Nash equilibrium? :",nash_equilibrium(id_to_strategy(0)))

    # checking club internal stability (nobody wants to quit)
    internal_stability = True
    for i in club:
        if i in ind_dev:
            internal_stability = False
    if not internal_stability:
        print(nb_tabs*"\t"+"Club is not internally stable (at least one member of the club wants to deviate)")
    
    for i in club:
        print(nb_tabs*"\t"+"%s gains %s from being in the club"%(i,rew[i] - rew_0[i]))
    for new_c in new_c_dev:
        explore_deviations(club + new_c,nb_tabs+1)
    
    return None

for c in c_dev:
    explore_deviations(c)"""