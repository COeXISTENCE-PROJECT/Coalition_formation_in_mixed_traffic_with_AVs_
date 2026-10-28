import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as lines

# ============================================================
# Matplotlib style
# ============================================================

mpl.rcParams.update(mpl.rcParamsDefault)

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "stix"

tnrfont = {"fontname": "Times New Roman"}

# ============================================================
# Bigger figure / font settings
# ============================================================

FIG_SCALE = 1.2

TITLE_SIZE = 24
LABEL_SIZE = 24
TICK_SIZE = 20
LEGEND_SIZE = 24

MARKER_SIZE = 12
LEGEND_MARKER_SIZE = 16

ARROW_WIDTH = 0.2
ARROW_HEAD_WIDTH = 0.6
ARROW_HEAD_LENGTH = 0.2

GRID_LINE_WIDTH = 1.0
REFERENCE_LINE_WIDTH = 1.4
LEGEND_LINE_WIDTH = 4

# ============================================================
# Helpers
# ============================================================

def strategy_to_id(s):
    """Convert joint action list like [1,1,0,...] to decimal id."""
    return int("".join(str(int(x)) for x in s), 2)


def get_humans(df):
    """Return only human rows, robust to 'Human'/'human' spelling."""
    return df[df["kind"].astype(str).str.lower() == "human"].copy()


def get_avs(df):
    """Return only AV rows, robust to 'AV'/'av' spelling."""
    return df[df["kind"].astype(str).str.lower() == "av"].copy()


def read_distinct_human_actions(base_folder="."):
    pattern = os.path.join(base_folder, "training_records*", "episodes", "ep1.csv")
    files = sorted(glob.glob(pattern))

    seen = {}

    print("Total ep1 files found:", len(files))

    for file in files:
        df = pd.read_csv(file)
        df = df.dropna(how="all")

        humans = get_humans(df)

        if humans.empty:
            continue

        humans = humans.sort_values("id").reset_index(drop=True)

        human_action = tuple(humans["action"].astype(int).tolist())

        if human_action in seen:
            continue

        seen[human_action] = {
            "file": file,
            "humans": humans,
            "human_action": human_action,
        }

    records = list(seen.values())

    # Put [0,0,0,0,0] first if it exists.
    records = sorted(
        records,
        key=lambda r: (
            r["human_action"] != (0, 0, 0, 0, 0),
            r["human_action"]
        )
    )

    return records


def print_distinct_human_joint_actions(base_folder="."):
    records = read_distinct_human_actions(base_folder=base_folder)

    print("\nDistinct human joint actions:", len(records))

    for r in records:
        humans = r["humans"]
        print("\nFile:", r["file"])
        print("Human IDs:", humans["id"].astype(int).tolist())
        print("Human joint action:", list(r["human_action"]))


# ============================================================
# Main plot
# ============================================================

def plot_human_actions(records, output_folder="../imgs"):
    os.makedirs(output_folder, exist_ok=True)

    if len(records) == 0:
        print("No distinct human actions found.")
        return

    # Find the baseline case: all humans on route 0.
    baseline_record = None

    for r in records:
        if r["human_action"] == (0, 0, 0, 0, 0):
            baseline_record = r
            break

    if baseline_record is None:
        raise ValueError("No baseline case found for human action (0, 0, 0, 0, 0).")

    baseline_humans = baseline_record["humans"].sort_values("id").reset_index(drop=True)
    baseline_times = baseline_humans["travel_time"].values

    nb_plots = len(records)
    ncols = 8
    nrows = int(np.ceil(nb_plots / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIG_SCALE * 2.6 * ncols, FIG_SCALE * 3.8 * nrows),
        sharey=True,
        gridspec_kw={
            "wspace": 0.08,
            "hspace": 0.35
        }
    )

    axes = np.array(axes).reshape(-1)

    for n, record in enumerate(records):
        ax = axes[n]

        humans = record["humans"].sort_values("id").reset_index(drop=True)
        human_action = record["human_action"]

        ids = humans["id"].astype(int).tolist()
        travel_times = humans["travel_time"].values
        actions = humans["action"].astype(int).tolist()

        ax.set_ylim(0.6, 1.3)

        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(ids, **tnrfont, fontsize=TICK_SIZE)

        # Keep x tick marks and human ID tick labels visible on all subplots.
        ax.tick_params(
            axis="x",
            which="major",
            bottom=True,
            labelbottom=True,
            length=6,
            width=1.2,
            direction="out"
        )

        # Show only the x-axis title "Human ID" on the lowest occupied row.
        lowest_row_start = ((nb_plots - 1) // ncols) * ncols

        if n >= lowest_row_start:
            ax.set_xlabel("Human ID", **tnrfont, size=LABEL_SIZE)
        else:
            ax.set_xlabel("")

        for label in ax.get_yticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(TICK_SIZE)

        ax.grid(
            axis="x",
            which="major",
            zorder=0,
            linewidth=GRID_LINE_WIDTH,
            alpha=0.6
        )

        for i, travel_time in enumerate(travel_times):

            if n != 0:
                diff = travel_time - baseline_times[i]

                # Higher travel time = worse.
                color = "r" if diff > 0 else "0"

                ax.annotate(
                    "",
                    xy=(i, travel_time),
                    xytext=(i, baseline_times[i]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=4,
                        mutation_scale=22,
                        alpha=0.85,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=2,
                )

            # Filled dot = route 0, white dot = route 1.
            if actions[i] == 0:
                ax.plot(
                    i,
                    travel_time,
                    "o",
                    markerfacecolor="0",
                    markeredgecolor="0",
                    markersize=MARKER_SIZE,
                    zorder=3,
                )
            else:
                ax.plot(
                    i,
                    travel_time,
                    "o",
                    markerfacecolor="1",
                    markeredgecolor="0",
                    markersize=MARKER_SIZE,
                    zorder=3,
                )

        if human_action == (0, 0, 0, 0, 0):
            title = "No human deviation"
        else:
            title = str(list(human_action))

        ax.set_title(title, **tnrfont, size=TITLE_SIZE, pad=8)
        if n >= lowest_row_start:
            ax.set_xlabel("Human ID", **tnrfont, size=LABEL_SIZE)
        else:
            ax.set_xlabel("")

        ax.set_box_aspect(1)

        # Show y-axis only on the leftmost subplot of each row.
        if n % ncols == 0:
            ax.yaxis.set_visible(True)
            ax.set_ylabel("Travel time", **tnrfont, size=LABEL_SIZE)

            for label in ax.get_yticklabels():
                label.set_fontname("Times New Roman")
                label.set_fontsize(TICK_SIZE)
        else:
            ax.yaxis.set_visible(False)
            ax.set_ylabel("")

    # Hide unused axes, if any.
    for k in range(nb_plots, len(axes)):
        axes[k].axis("off")

    axes[0].set_ylabel("Travel time", **tnrfont, size=LABEL_SIZE)
    axes[0].set_facecolor("aliceblue")

    # ========================================================
    # Legend
    # ========================================================

    black_dot = lines.Line2D(
        [0], [0],
        color="w",
        marker="o",
        markerfacecolor="0",
        markeredgecolor="0",
        markersize=LEGEND_MARKER_SIZE
    )

    white_dot = lines.Line2D(
        [0], [0],
        color="w",
        marker="o",
        markerfacecolor="1",
        markeredgecolor="0",
        markersize=LEGEND_MARKER_SIZE
    )

    black_arrow = lines.Line2D(
        [0], [0],
        color="r",
        linewidth=LEGEND_LINE_WIDTH
    )

    red_arrow = lines.Line2D(
        [0], [0],
        color="b",
        linewidth=LEGEND_LINE_WIDTH
    )

    fig.legend(
        [black_dot, white_dot, black_arrow, red_arrow],
        ["Human on route 0", "Human on route 1", "Gained / no loss", "Lost"],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.96),
        frameon=False,
        fontsize=LEGEND_SIZE
    )

    for text in fig.legends[0].get_texts():
        text.set_fontname("Times New Roman")

    plt.tight_layout(
        rect=[0, 0, 1, 0.92],
        h_pad=4.0,
        w_pad=1.4
    )

    # ========================================================
    # Save
    # ========================================================

    png_path = os.path.join(output_folder, "human_joint_action_travel_times.png")
    svg_path = os.path.join(output_folder, "human_joint_action_travel_times.svg")
    pdf_path = os.path.join(output_folder, "human_joint_action_travel_times.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.show()

    print("Saved:", png_path)
    print("Saved:", svg_path)
    print("Saved:", pdf_path)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    records = read_distinct_human_actions(base_folder="0_1_5_4_human_deviations_training_records")

    print("\nDistinct human joint actions:", len(records))

    for r in records:
        print(r["human_action"], "from", r["file"])

    plot_human_actions(records, output_folder="../imgs")