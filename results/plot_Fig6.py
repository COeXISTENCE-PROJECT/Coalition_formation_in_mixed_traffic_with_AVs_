import pandas as pd
import matplotlib.pyplot as plt

list = ["alltraveltimes_noclub.csv","alltraveltimes_club_1_5_6.csv","alltraveltimes_club_1_5_6_7.csv","alltraveltimes_club_0_1_5_6_7.csv","alltraveltimes_club_0_1_5_6.csv"]

def filter(tab, column, kind):
    return tab[tab[column] == kind]

df_x0 = pd.read_csv("alltraveltimes_noclub.csv")[["travel_time","action","kind"]]

AVG = []
HDV = []
CDV = []
R0 = []
R1 = []

for filename in list:
    df = pd.read_csv(filename)[["travel_time","action","kind"]]
    df["normalized_traveltime"] = df["travel_time"]/df_x0["travel_time"]
    dfCDV,dfHDV = filter(df,"kind","AV"), filter(df,"kind","Human")
    dfR0,dfR1 = filter(df,"action",0), filter(df,"action",1)

    AVG.append(df["normalized_traveltime"].mean())
    HDV.append(dfHDV["normalized_traveltime"].mean())
    CDV.append(dfCDV["normalized_traveltime"].mean())
    R0.append(dfR0["normalized_traveltime"].mean())
    R1.append(dfR1["normalized_traveltime"].mean())

    print(df["normalized_traveltime"].mean(), dfHDV["normalized_traveltime"].mean(), dfCDV["normalized_traveltime"].mean())
    print(df["normalized_traveltime"].mean(), dfR0["normalized_traveltime"].mean(), dfR1["normalized_traveltime"].mean())
    print()


# -------------------------------------------------------------------------
# Journal-style bar plots
# -------------------------------------------------------------------------

import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

tnrfont = {"fontname": "Times New Roman"}

fig = plt.figure(figsize=(7.2, 4), constrained_layout=False)

gs = fig.add_gridspec(
    nrows=1,
    ncols=2,
    width_ratios=[1, 1],
    wspace=0.28,
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

colors = {
    "all": "firebrick",
    "human": "teal",
    "av": "peru",
    "route0": "teal",
    "route1": "peru",
    "initial": "0",
}

width = 0.16
x = np.arange(2)

# -------------------------------------------------------------------------
# a) Vehicle travel times
# -------------------------------------------------------------------------

ax1.bar(
    x - width,
    [AVG[0], AVG[1]],
    label="All drivers",
    color=colors["all"],
    width=width,
    zorder=3,
)

ax1.bar(
    x,
    [HDV[0], HDV[1]],
    label="Humans",
    color=colors["human"],
    width=width,
    zorder=3,
)

ax1.bar(
    x + width,
    [CDV[0], CDV[1]],
    label="AVs",
    color=colors["av"],
    width=width,
    zorder=3,
)

ax1.axhline(
    AVG[0],
    color=colors["initial"],
    linewidth=2,
    label="Initial situation",
    zorder=2,
)

ax1.axhline(0, color="0", linewidth=2, zorder=2)

ax1.set_xticks(x)
ax1.set_xticklabels(["No club", "[1,5,6]"], **tnrfont)

ax1.set_xlim(-0.45, 1.45)
ax1.set_ylim(0, 1.7)
ax1.set_yticks(np.arange(0, 1.8, 0.2))

ax1.set_xlabel("Members of the coalition", **tnrfont)
ax1.set_ylabel("Average normalized travel time", **tnrfont)
ax1.set_title("a) Vehicle travel times", loc="center", pad=7, **tnrfont)

ax1.set_axisbelow(True)
ax1.grid(axis="y", which="major", color="0.90", linewidth=0.6)

ax1.tick_params(direction="out", top=False, right=False)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

leg1 = ax1.legend(
    loc="upper left",
    ncols=2,
    frameon=False,
    handlelength=1.2,
    columnspacing=0.6,
    handletextpad=0.35,
    borderaxespad=0.2,
    prop={"family": "Times New Roman", "size": 12},
)
# -------------------------------------------------------------------------
# b) Travel times on each route
# -------------------------------------------------------------------------

ax2.bar(
    x - width,
    [AVG[0], AVG[1]],
    label="All drivers",
    color=colors["all"],
    width=width,
    zorder=3,
)

ax2.bar(
    x,
    [R0[0], R0[1]],
    label="Route 0",
    color=colors["route0"],
    width=width,
    zorder=3,
)

ax2.bar(
    x + width,
    [R1[0], R1[1]],
    label="Route 1",
    color=colors["route1"],
    width=width,
    zorder=3,
)

ax2.axhline(
    AVG[0],
    color=colors["initial"],
    linewidth=2,
    label="Initial situation",
    zorder=2,
)

ax2.axhline(0, color="0", linewidth=2, zorder=2)

ax2.set_xticks(x)
ax2.set_xticklabels(["No club", "[1,5,6]"], **tnrfont)

ax2.set_xlim(-0.45, 1.45)
ax2.set_ylim(0, 1.7)
ax2.set_yticks(np.arange(0, 1.8, 0.2))

ax2.set_xlabel("Members of the coalition", **tnrfont)
ax2.set_title("b) Travel times on each route", loc="center", pad=7, **tnrfont)

ax2.set_axisbelow(True)
ax2.grid(axis="y", which="major", color="0.90", linewidth=0.6)

ax2.tick_params(direction="out", top=False, right=False)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

leg2 = ax2.legend(
    loc="upper left",
    ncols=2,
    frameon=False,
    handlelength=1.2,
    columnspacing=0.6,
    handletextpad=0.35,
    borderaxespad=0.2,
    prop={"family": "Times New Roman", "size": 12},
)
# -------------------------------------------------------------------------
# Final layout and export
# -------------------------------------------------------------------------

fig.subplots_adjust(
    left=0.065,
    right=0.995,
    bottom=0.20,
    top=0.86,
    wspace=0.22,
)

os.makedirs("../imgs", exist_ok=True)

plt.savefig("../imgs/average_travel_times_bars.pdf", bbox_inches="tight", dpi=600)
plt.savefig("../imgs/average_travel_times_bars.svg", bbox_inches="tight", pad_inches=0.02)
plt.savefig("../imgs/average_travel_times_bars.png", bbox_inches="tight",   dpi=600)
plt.savefig("../imgs/average_travel_times_bars_transparent.png", bbox_inches="tight",   dpi=600, transparent=True)
plt.savefig("../imgs/average_travel_times_bars_transparent.svg", bbox_inches="tight",   transparent=True)

plt.show()



