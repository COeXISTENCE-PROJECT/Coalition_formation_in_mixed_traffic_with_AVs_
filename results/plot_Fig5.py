import os
import pandas as pd
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.lines as lines
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch

viridis = mpl.cm.viridis
viridis_r = mpl.cm.viridis_r
norm = mpl.colors.Normalize(vmin=5, vmax=10)


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

def custom_df_human13(tl_list, custom_filename="reward_df_custom_human13.csv"):
    assert len(tl_list) == 11

    with open(custom_filename, "w") as f:
        f.write("id,0,1,2,3,4,5,6,7,8,9\n")

        for id in range(1024):
            s = id_to_strategy(id)
            n_1 = sum(s)

            tl_0, tl_1, tl_y, nb_agents = tl_list[n_1]

            file_name = (
                f"reward_df_{tl_0}_{tl_1}_{tl_y}_{nb_agents}"
                f"agents_human_13_deviates.csv"
            )

            df_reward = pd.read_csv(file_name)

            text = str(id)
            for i in range(10):
                text += "," + str(float(df_reward[str(i)].values[id]))

            f.write(text + "\n")


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


def travel_times(id):
    s = id_to_strategy(id)
    t0 = 0
    t1 = 0
    n0 = 0
    n1 = 0
    les_t = tab_reward[:,id]
    for i in range(10):
        if s[i] == 0:
            n0 += 1
            t0 += les_t[i]
        else:
            n1 += 1
            t1 += les_t[i]
    if n0 > 0:
        t0 = t0/n0
    if n1 > 0:
        t1 = t1/n1
    return n0, n1, -t0, -t1

def normalized_travel_times(id,p0):
    s = id_to_strategy(id)
    t0 = 0
    t1 = 0
    n0 = 0
    n1 = 0
    les_t = tab_reward[:,id]
    for i in range(10):
        if s[i] == 0:
            n0 += 1
            t0 += les_t[i]/p0[i]
        else:
            n1 += 1
            t1 += les_t[i]/p0[i]
    if n0 > 0:
        t0 = t0/n0
    if n1 > 0:
        t1 = t1/n1
    return n0, n1, t0, t1

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


def plot_deviation(club, p0):

    '''
    Input : `club` is a subset of the set of players.
    Output : `y` is the normalized reward/travel-time ratio of all players
             if the club goes to route 1 and all other players go to route 0.
             `y_dev` is the normalized payoff change if each player deviates alone.
    '''

    s = coalition_to_strategy(club)

    print("#######################")
    print("Club is %s. Joint action is %s" % (club, s))

    p = s_to_reward(s)
    nash_dev = nash_deviation(s)

    # Raw travel times are the negative of payoffs
    raw_travel_times = [-p[i] for i in range(len(p))]

    print("Raw travel times:")
    for i, tt in enumerate(raw_travel_times):
        route = s[i]
        print("  Agent %s on route %s: travel time = %.4f" % (i, route, tt*60))

    print("List of agents that could deviate individually: \t", individual_deviations(s))
    print("List of coalitions that could deviate simultaneously: \t", coalition_deviations(s))
    print(
        "Nash equilibrium: %s, strong equilibrium: %s"
        % (nash_equilibrium(s), strong_nash_equilibrium(s, verbose=False))
    )

    # Keep this as before, because the plot uses normalized travel times
    y = [p[i] / p0[i] for i in range(len(p))]
    dev = [nash_dev[i] / p0[i] for i in range(10)]

    return y, dev, s



# Define system parameters
nb_agents = 15      # number of human + AV agents in the network
threshold = 2       # number of AVs on route 1 beyond which the traffic lights adapt
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
custom_df_human13(tl_list, "reward_df_custom_human13.csv")


tab_reward_base = build_rewardtable("reward_df_custom.csv")
tab_reward_h13 = build_rewardtable("reward_df_custom_human13.csv")
tab_reward = tab_reward_base

# Calculating deviations from the initial situation
s_0 = id_to_strategy(0)
print("#######################\nInitial situation: everyone on route 0.")
print("List of clubs that may form from x^0: \t",coalition_deviations(s_0))
print("Nash equilibrium: %s, strong equilibrium: %s"%(nash_equilibrium(s_0),strong_nash_equilibrium(s_0,verbose=False)))
p_0 = s_to_reward(s_0)
nash_dev = nash_deviation(s_0)


print("Nash equilbrium: %s, strong equilibrium: %s"%(nash_equilibrium(s_0),strong_nash_equilibrium(s_0,verbose=False)))


# Preparing the plot
les_x = [i for i in range(10)]
plt.style.use("default")
tnrfont = {'fontname':'Times New Roman'}
# plt.style.use('science')
fig = plt.figure()
fig.set_size_inches(20,4)
list_clubs = [[],[1,5,6],[1,5,6,7],[0,1,5,6,7],[0,1,5,6],[0,1,5,6]]
nb_clubs_tested = len(list_clubs)
les_ax = []

les_xy = []

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
        les_ax[-1].arrow(i+shift,les_y[i],0,les_dev[i],width=0.15,head_width = 0.45, head_length = min(0.1,np.abs(les_dev[i])),length_includes_head = True,color=color,zorder=2,alpha = 0.8)
        if s[i] == 0:
            les_ax[-1].plot(i+shift,les_y[i],"o",markerfacecolor='0',markeredgecolor='0',markersize=10,zorder=3)
        else:
            les_ax[-1].plot(i+shift,les_y[i],"o",markerfacecolor='1',markeredgecolor='0',markersize=10,zorder=3)
        plt.draw()
        # ax.plot([i + shift,i + shift],les_yplusdev[i],color=color,alpha=0.5)

for n in range(nb_clubs_tested):
    club = list_clubs[n]

    if n == 5:
        tab_reward = tab_reward_h13
    else:
        tab_reward = tab_reward_base

    les_ax.append(fig.add_subplot(1,nb_clubs_tested,n+1))
    les_ax[-1].set_ylim(0.8,3)
    les_ax[-1].set_xticks([i  for i in range(10)], minor=False)
    les_ax[-1].set_yticks([0.8,1,1.5,2,2.5,3], minor = False)
    les_ax[-1].set_xticklabels(les_ax[-1].get_xticks(), **tnrfont, size=16)
    les_ax[-1].set_yticklabels(les_ax[-1].get_yticks(), **tnrfont, size=16)
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
les_ax[0].yaxis.set_visible(True)
les_ax[0].set_ylabel("Normalized travel time",**tnrfont, size=20)
les_ax[0].set_facecolor('aliceblue')
les_ax[4].set_facecolor('honeydew')

les_ax[0].set_title("a) No club",**tnrfont,size=20)
les_ax[1].set_title("b) Club is [1,5,6]",**tnrfont,size=20)
les_ax[2].set_title("c) Club is [1,5,6,7]",**tnrfont,size=20)
les_ax[3].set_title("d) Club is [0,1,5,6,7]",**tnrfont,size=20)
les_ax[4].set_title("e) Club is [0,1,5,6]",**tnrfont,size=20)


for ax in les_ax:
    ax.tick_params(width=1.5)


for ax in les_ax:
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        
les_ax[5].set_title(
    "f) Club is [0,1,5,6]",
    **tnrfont,
    size=20
)
les_ax[5].set_facecolor('mistyrose')


les_ax[0].set_xlabel("Agent ID", **tnrfont, size=20)
les_ax[1].set_xlabel("Agent ID", **tnrfont, size=20)
les_ax[2].set_xlabel("Agent ID", **tnrfont, size=20)
les_ax[3].set_xlabel("Agent ID", **tnrfont, size=20)
les_ax[4].set_xlabel("Agent ID", **tnrfont, size=20)
les_ax[5].set_xlabel("Agent ID", **tnrfont, size=20)

black_dot = lines.Line2D([0], [0], color = "w", marker="o",markerfacecolor='0',markeredgecolor='0',markersize=14)
white_dot = lines.Line2D([0], [0], color = "w", marker="o",markerfacecolor='1',markeredgecolor='0',markersize=14)

# leg = plt.legend([black_dot, white_dot], ["Agent on route 0", "Agent on route 1"],loc = 'upper right')

# plt.title("(%s,%s) below threshold, (%s,%s) above threshold, threshold =  %s"%(tl_0_below_threshold,tl_1_below_threshold,tl_0_above_threshold,tl_1_above_threshold,threshold))
plt.savefig("..\imgs\deviations.png", dpi=300)
plt.savefig("..\imgs\deviations.svg")
plt.show()