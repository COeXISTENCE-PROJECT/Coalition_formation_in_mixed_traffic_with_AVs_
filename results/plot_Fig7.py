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

fig = plt.figure()
fig.set_size_inches(10,4)
colors = ["firebrick", "teal", "peru", "navy", "salmon", "slategray", "darkviolet", "lightskyblue", "darkolivegreen", "black"]
tnrfont = {'fontname':'Times New Roman'}

width = 0.16

## 1st plot

ax = fig.add_subplot(121)
ax.bar([i - width for i in range(2)],[AVG[0],AVG[1]], label = "All drivers", color=colors[0],width=width)
ax.bar([i for i in range(2)],[HDV[0],HDV[1]], label = "Humans", color=colors[1],width=width)
ax.bar([i + width for i in range(2)],[CDV[0],CDV[1]], label = "AVs", color=colors[2],width=width)

ax.set_xticks([0,1])
ax.set_xlim(-0.5,1.5)
ax.set_ylim(0,1.7)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6])
ax.axhline(AVG[0], color='0', linewidth=1, label="Initial situation")
ax.axhline(0, color='0', linewidth=1)
ax.grid(axis="x",which="major",zorder=0)

ax.set_xticklabels(["No club","[0,1,5,6]"], **tnrfont, size=12)
ax.set_yticklabels(ax.get_yticks(), **tnrfont, size=12)

ax.set_xlabel("Members of the coalition", **tnrfont, size=12)
ax.set_ylabel("Average of normalized travel times", **tnrfont,size=15)
ax.set_title("a) Vehicle travel times", **tnrfont, size=15)

ax.legend(prop={'family':tnrfont['fontname'], 'size':12}, loc='upper center', bbox_to_anchor=(0.5,1), ncol=2, frameon=True)

## 2nd plot

ax = fig.add_subplot(122)
ax.bar([i - width for i in range(2)],[AVG[0],AVG[1]], label= "All drivers", color=colors[0],width=width)
ax.bar([i for i in range(2)],[R0[0],R0[1]], label = "Route 0", color=colors[1],width=width)
ax.bar([i + width for i in range(2)],[R1[0],R1[1]], label = "Route 1", color=colors[2],width=width)

ax.set_xticks([0,1])
ax.set_xlim(-0.5,1.5)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6])
ax.set_ylim(0,1.7)
ax.axhline(AVG[0], color='0', linewidth=1, label="Initial situation")
ax.axhline(0, color='0', linewidth=1)
ax.grid(axis="x",which="major",zorder=0)

ax.set_xticklabels(["No club","[1,5,6]"], **tnrfont, size=12)
ax.set_yticklabels(ax.get_yticks(), **tnrfont, size=12)
ax.set_xlabel("Members of the coalition", **tnrfont, size=12)
ax.set_title("b) Travel times on each route", **tnrfont, size=15)
# ax.set_ylabel("Average of normalized travel times", **tnrfont)

ax.legend(prop={'family':tnrfont['fontname'], 'size':12}, loc='upper center', bbox_to_anchor=(0.5,1), ncol=2, frameon=True)

plt.draw()
plt.savefig("../imgs/average_travel_times_bars.png")
plt.savefig("../imgs/average_travel_times_bars.svg")
plt.savefig("../imgs/average_travel_times_bars_transparent.png", transparent=True)
plt.savefig("../imgs/average_travel_times_bars_transparent.svg", transparent=True)
plt.show()