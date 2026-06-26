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
tab_reward = build_rewardtable("reward_df_custom.csv")

# Calculating deviations from the initial situation
s_0 = id_to_strategy(0)
print("#######################\nInitial situation: everyone on route 0.")
print("List of clubs that may form from x^0: \t",coalition_deviations(s_0))
print("Nash equilibrium: %s, strong equilibrium: %s"%(nash_equilibrium(s_0),strong_nash_equilibrium(s_0,verbose=False)))
p_0 = s_to_reward(s_0)
nash_dev = nash_deviation(s_0)


## Compute Nash equilibria and deviations

les_t0 = [[] for n0 in range(11)]
les_t1 = [[] for n0 in range(11)]

snash_x, snash_y = [], []
nash_x, nash_y = [], []
notnash_x, notnash_y = [], []

clubs_snash_x, clubs_snash_y = [], []
clubs_nash_x, clubs_nash_y = [], []
clubs_notnash_x, clubs_notnash_y = [], []

list_clubs = [[5, 6],
 [5, 6, 9],
 [5, 6, 7, 9],
 [3, 5, 6, 7, 9],
 [1, 3, 5, 6, 7, 9],
 [0, 1, 3, 5, 6, 7, 9],
 [0, 1, 3, 4, 5, 6, 7, 9],
 [0, 3, 5, 6, 7, 9],
 [0, 3, 4, 5, 6, 7, 9],
 [3, 4, 5, 6, 7, 9],
 [1, 3, 4, 5, 6, 7, 9],
 [1, 5, 6, 7, 9],
 [0, 1, 5, 6, 7, 9],
 [0, 1, 4, 5, 6, 7, 9],
 [1, 4, 5, 6, 7, 9],
 [0, 5, 6, 7, 9],
 [0, 4, 5, 6, 7, 9],
 [4, 5, 6, 9],
 [1, 4, 5, 6, 9],
 [0, 1, 4, 5, 6, 9],
 [1, 3, 4, 5, 6, 9],
 [0, 1, 3, 4, 5, 6, 9],
 [0, 4, 5, 6, 9],
 [0, 3, 4, 5, 6, 9],
 [3, 5, 6, 9],
 [3, 4, 5, 6, 9],
 [1, 3, 5, 6, 9],
 [0, 1, 3, 5, 6, 9],
 [0, 3, 5, 6, 9],
 [1, 5, 6, 9],
 [0, 1, 5, 6, 9],
 [0, 5, 6, 9],
 [5, 6, 7],
 [4, 5, 6, 7],
 [3, 4, 5, 6, 7],
 [1, 3, 4, 5, 6, 7],
 [0, 1, 3, 4, 5, 6, 7],
 [0, 3, 4, 5, 6, 7],
 [1, 4, 5, 6, 7],
 [0, 1, 4, 5, 6, 7],
 [0, 4, 5, 6, 7],
 [3, 5, 6, 7],
 [1, 3, 5, 6, 7],
 [0, 1, 3, 5, 6, 7],
 [0, 3, 5, 6, 7],
 [1, 5, 6, 7],
 [0, 1, 5, 6, 7],
 [0, 5, 6, 7],
 [3, 5, 6],
 [3, 4, 5, 6],
 [1, 3, 4, 5, 6],
 [0, 1, 3, 4, 5, 6],
 [0, 3, 4, 5, 6],
 [1, 3, 5, 6],
 [0, 1, 3, 5, 6],
 [0, 3, 5, 6],
 [1, 5, 6],
 [0, 1, 5, 6],
 [0, 1, 4, 5, 6],
 [1, 4, 5, 6],
 [0, 5, 6],
 [0, 4, 5, 6],
 [1, 6],
 [1, 6, 9],
 [0, 1, 6, 9],
 [0, 1, 4, 6, 9],
 [0, 1, 6],
 [0, 1, 4, 6],
 [1, 5],
 [1, 5, 9],
 [1, 5, 7, 9],
 [0, 1, 5, 7, 9],
 [0, 1, 4, 5, 7, 9],
 [0, 1, 5, 9],
 [0, 1, 4, 5, 9],
 [1, 5, 7],
 [0, 1, 5, 7],
 [0, 1, 4, 5, 7],
 [0, 1, 5],
 [0, 1, 4, 5]]

# list_clubs = [[4,5,9]]
list_clubs_ids = [strategy_to_id(coalition_to_strategy(c)) for c in list_clubs]

rew = id_to_reward(0)
list_c = []

# Store labels only for actions that will get a black circle
black_circle_label_x = []
black_circle_label_y = []
black_circle_labels = []



for id in range(1024):
    # n0,n1,t0,t1 = travel_times(id)
    n0,n1,t0,t1 = normalized_travel_times(id,p_0)
    les_t0[n0].append(t0)
    les_t1[n0].append(t1)

    if n0*n1 > 0: # we remove the dots (n0,n1) = (0,10) and (n0,n1) = (10,0)

        s = id_to_strategy(id)

        if nash_equilibrium(id_to_strategy(id)):
            if strong_nash_equilibrium(id_to_strategy(id),verbose=False):
                snash_x.append(t0)
                snash_y.append(t1)
            else:
                nash_x.append(t0)
                nash_y.append(t1)
        else:
            notnash_x.append(t0)
            notnash_y.append(t1)
        
        """if id in list_clubs_ids:
            if nash_equilibrium(id_to_strategy(id)):
                if strong_nash_equilibrium(id_to_strategy(id),verbose=False):
                    print("\n\n\n It is a strong nash equilibrium with id: ", id, "\n\n\n")
                    clubs_snash_x.append(t0)
                    clubs_snash_y.append(t1)
                else:
                    clubs_nash_x.append(t0)
                    clubs_nash_y.append(t1)
            else:
                clubs_notnash_x.append(t0)
                clubs_notnash_y.append(t1)"""
        if id in list_clubs_ids:

            joint_action = id_to_strategy(id)
            joint_action_label = "".join(map(str, joint_action))

            print(
                f"Black circle action | "
                f"id = {id} | "
                f"joint action = {joint_action} | "
                f"label = {joint_action_label} | "
                f"point = ({t0:.3f}, {t1:.3f})"
            )

            # Save label position for plotting
            black_circle_label_x.append(t0)
            black_circle_label_y.append(t1)
            black_circle_labels.append(joint_action_label)

            if nash_equilibrium(s):
                if strong_nash_equilibrium(s, verbose=False):
                    print("\n\n\n It is a strong nash equilibrium with id: ", id, "\n\n\n")
                    clubs_snash_x.append(t0)
                    clubs_snash_y.append(t1)
                else:
                    clubs_nash_x.append(t0)
                    clubs_nash_y.append(t1)
            else:
                clubs_notnash_x.append(t0)
                clubs_notnash_y.append(t1)


# Drawing plots

# Drawing plots -- journal style
# This block keeps the original colour choices, but improves layout, typography,
# spacing, line widths, legend placement, and export quality.

# Journal-friendly typography and vector export settings
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 18,
    "axes.titlesize": 24,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "axes.linewidth": 2,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

tnrfont = {"fontname": "Times New Roman"}

# Wide format suitable for a top-of-page figure.
# For a two-column IEEE-style page, you may prefer figsize=(7.16, 3.05).
fig = plt.figure(figsize=(14.8, 3.8), constrained_layout=False)

gs = fig.add_gridspec(
    nrows=1,
    ncols=5,
    width_ratios=[1.45, 0.04, 0.10, 1.45, 0.42],
    wspace=0.08,
)

ax1 = fig.add_subplot(gs[0, 0])
cbar_ax = fig.add_subplot(gs[0, 1])
spacer_ax = fig.add_subplot(gs[0, 2])
ax2 = fig.add_subplot(gs[0, 3])
leg_ax = fig.add_subplot(gs[0, 4])

spacer_ax.axis("off")
leg_ax.axis("off")


# Common axis formatting
for ax in (ax1, ax2):
    ax.set_aspect("auto")

    # Do not set fixed x/y limits.
    # Use automatic limits, but generate ticks/grid lines across the full range.
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))

    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))

    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="0.90", linewidth=0.5)

    ax.tick_params(direction="out", top=False, right=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# -------------------------------------------------------------------------
# (a) Travel times and traffic flow
# -------------------------------------------------------------------------
for n0 in range(1, 10):
    ax1.scatter(
        les_t0[n0],
        les_t1[n0],
        s=150,
        alpha=0.5,
        color=viridis(n0 / 10),
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

ax1.axvline(les_t0[10], linewidth=2, color="0", label="All AVs on Route 0", zorder=1)
ax1.axhline(les_t1[0], linewidth=2, linestyle="--", color="0", label="All AVs on Route 1", zorder=1)
ax1.axhline(1, linewidth=0.8, color="0.5", zorder=1)
# Equal-performance line, drawn over the automatically chosen axis range
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
lo = min(xlim[0], ylim[0])
hi = max(xlim[1], ylim[1])
ax1.plot([lo, hi], [lo, hi], linewidth=2, color="magenta", zorder=1)
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)

ax1.set_xlabel("Average normalized travel time on Route 0",  fontsize=22, **tnrfont)
ax1.set_ylabel(f"Average normalized \ntravel time on Route 1",  fontsize=22,**tnrfont)
ax1.set_title("a) Travel times", loc="center", pad=7, **tnrfont)
ax1.legend(
    loc="upper right",
    frameon=False,
    handlelength=3,
    borderaxespad=0.2,
    prop={"family": "Times New Roman", "size": 16},
)

# Colour bar for panel (a); colours are unchanged.
norm = mpl.colors.Normalize(vmin=0, vmax=10)
sm = mpl.cm.ScalarMappable(cmap=viridis_r, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")

cbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
cbar.ax.tick_params(labelsize=12, length=2.5, pad=2)
cbar.outline.set_linewidth(0.6)

# Put colorbar ticks and label on the LEFT side,
# away from the y-axis labels/ticks of panel (b).
cbar_ax.yaxis.set_ticks_position("left")
cbar_ax.yaxis.set_label_position("left")

cbar.set_label(
    "Number of vehicles on Route 1",
    fontsize=15,  
    labelpad=8,
    rotation=90,
    va="center",
    **tnrfont,
)


# Make the colorbar thinner without changing the legend area
"""pos = cbar_ax.get_position()
new_height = pos.height * 0.35
cbar_ax.set_position([
    pos.x0,
    pos.y0 + 0.5 * (pos.height - new_height),
    pos.width,
    new_height
])"""

# -------------------------------------------------------------------------
# (b) Nash and strong equilibria
# -------------------------------------------------------------------------
ax2.scatter(
    notnash_x,
    notnash_y,
    label="Not a Nash equilibrium",
    s=150,
    alpha=0.5,
    color="firebrick",
    linewidths=0,
    rasterized=True,
    zorder=2,
)

ax2.scatter(
    nash_x,
    nash_y,
    label="Nash equilibrium but not strong",
    s=150,
    alpha=0.8,
    color="orange",
    linewidths=0,
    rasterized=True,
    zorder=3,
)
ax2.scatter(
    snash_x,
    snash_y,
    label="Nash and strong equilibrium",
    s=180,
    alpha=0.8,
    color="darkolivegreen",
    linewidths=0,
    rasterized=True,
    zorder=4,
)

# Identified clubs are plotted last so they remain visible.
ax2.scatter(
    clubs_notnash_x,
    clubs_notnash_y,
    alpha=1,
    s=180,
    color="firebrick",
    edgecolors="none",
    zorder=6,
)

ax2.scatter(
    clubs_nash_x,
    clubs_nash_y,
    alpha=1,
    s=180,
    color="1",
    edgecolors="none",
    zorder=7,
)

ax2.scatter(
    clubs_snash_x,
    clubs_snash_y,
    alpha=1,
    s=180,
    color="darkolivegreen",
    edgecolors="none",
    zorder=11,
)

# Black outline ONLY around identified-club actions
# that are also Nash and strong
ax2.scatter(
    clubs_snash_x,
    clubs_snash_y,
    s=180,
    facecolors="none",
    edgecolors="black",
    linewidths=1.5,
    zorder=13,
)

# Strong black outline around ALL identified-club actions
clubs_all_x = clubs_notnash_x + clubs_nash_x + clubs_snash_x
clubs_all_y = clubs_notnash_y + clubs_nash_y + clubs_snash_y

ax2.scatter(
    clubs_all_x,
    clubs_all_y,
    s=180,
    facecolors="none",
    edgecolors="black",
    linewidths=0.8,
    label="Action with an identified club",
    zorder=10,
)

ax2.axvline(les_t0[10], linewidth=2, color="0", zorder=1)
ax2.axhline(1, linewidth=2, color="0.5", zorder=1)
ax2.axhline(les_t1[0], linewidth=2, linestyle="--", color="0", zorder=1)
# Equal-performance line, drawn over the automatically chosen axis range
xlim = ax2.get_xlim()
ylim = ax2.get_ylim()
lo = min(xlim[0], ylim[0])
hi = max(xlim[1], ylim[1])
ax2.plot(
    [lo, hi],
    [lo, hi],
    linewidth=2,
    label=f"Equal performance between \nthe two routes",
    color="magenta",
    zorder=1,
)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.set_xlabel("Average normalized travel time on Route 0",  fontsize=22,**tnrfont)
ax2.set_title("b) Equilibria", loc="center", pad=7, **tnrfont)

# Dedicated legend axis below panel (b), rather than pushing the plot downward.
handles, labels = ax2.get_legend_handles_labels()
leg = leg_ax.legend(
    handles,
    labels,
    loc="center left",
    ncols=1,
    frameon=False,
    handlelength=2.0,
    columnspacing=1.4,
    handletextpad=0.5,
    markerscale=1.8,   # makes legend circles bigger
    prop={"family": "Times New Roman", "size": 16},
)

for handle, label in zip(leg.legend_handles, labels):
    if hasattr(handle, "set_alpha"):
        handle.set_alpha(0.8)

    # Make "Action with an identified club" legend marker same size
    if label == "Action with an identified club" or label =="Nash and strong equilibrium" or label =="Nash equilibrium but not strong" or label =="Not a Nash equilibrium":
        handle.set_sizes([100 * 1.8**2])

fig.subplots_adjust(
    left=0.035,
    right=0.995,
    bottom=0.18,
    top=0.88,
)

# Move the heatmap/colorbar a bit to the right
pos = cbar_ax.get_position()
cbar_ax.set_position([
    pos.x0 + 0.02,
    pos.y0,
    pos.width,
    pos.height
])


# Export. PDF/SVG are best for journals; PNG is kept for quick preview.
os.makedirs("../imgs", exist_ok=True)
plt.savefig("../imgs/travel_times.pdf", bbox_inches="tight", dpi=600)
plt.savefig("../imgs/travel_times.svg", bbox_inches="tight")
plt.savefig("../imgs/travel_times.png", bbox_inches="tight", dpi=600)
plt.show()