"""
AV-only coalition analysis for your 2048 joint-action system.

This script does two things:

1. Builds reward_df_custom_human13.csv from:
       training_records/ep0.csv
       training_records/ep1.csv
       ...
       training_records/ep2047.csv

2. Uses that payoff table to check which AV coalitions can form.

System:
    - 10 AVs: ids 0,1,...,9
    - 1 variable human: human 13
    - human 13 is NOT allowed to be part of coalitions
    - human 13's action is held fixed during coalition checks

Important:
    This script only needs AV payoffs to check AV coalitions.
    Human 13 affects the episode/payoffs, but human 13 does not need
    to be in the payoff table because humans cannot join coalitions.
"""

import itertools
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

NB_AVS = 10
AV_IDS = list(range(NB_AVS))

NB_JOINT_ACTIONS = 2 ** 11  # 10 AVs + 1 human = 2048

EPISODES_FOLDER = "training_records"
REWARD_CSV = "reward_df_custom_human13.csv"

# The original notebook uses reward = -travel_time.
# Keep this True if your episode CSV has travel_time and you want reward.
USE_NEGATIVE_TRAVEL_TIME_AS_REWARD = True

# If your final payoff table stores rewards/utilities, larger is better.
# Since reward = -travel_time, larger is better.
HIGHER_IS_BETTER = True


# ============================================================
# Build the missing 2048-row payoff table from ep*.csv files
# ============================================================

def find_episode_file(joint_action_id, episodes_folder=EPISODES_FOLDER):
    """
    Finds the CSV file for a given joint action.

    Your current script saves:
        training_records/ep0.csv
        training_records/ep1.csv
        ...
        training_records/ep2047.csv

    The original notebook sometimes used:
        training_records/episodes/ep1.csv
        ...
        training_records/episodes/ep1024.csv

    This function supports both styles.
    """
    folder = Path(episodes_folder)

    candidates = [
        folder / f"ep{joint_action_id}.csv",
        folder / "episodes" / f"ep{joint_action_id}.csv",
        folder / "episodes" / f"ep{joint_action_id + 1}.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find episode file for joint_action_id={joint_action_id}.\n"
        f"Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )


def episode_to_av_rewards(episode_csv):
    """
    Reads one episode CSV and extracts the 10 AV rewards.

    Expected episode columns from the original logic:
        kind
        start_time
        travel_time

    The original notebook did:
        df = df[df['kind'] == 'AV']
        df = df.sort_values(by='start_time')
        df['reward'] = -1 * df['travel_time']

    This function follows the same idea.
    """
    df = pd.read_csv(episode_csv)

    if "kind" not in df.columns:
        raise ValueError(
            f"{episode_csv} does not have a 'kind' column.\n"
            f"Available columns are: {list(df.columns)}"
        )

    av_df = df[df["kind"].astype(str).str.upper() == "AV"].copy()

    if len(av_df) != NB_AVS:
        raise ValueError(
            f"{episode_csv} should contain {NB_AVS} AV rows, "
            f"but it contains {len(av_df)} AV rows."
        )

    if "start_time" in av_df.columns:
        av_df = av_df.sort_values(by="start_time").reset_index(drop=True)
    else:
        av_df = av_df.reset_index(drop=True)

    if "reward" in av_df.columns:
        rewards = av_df["reward"].astype(float).tolist()
    elif "travel_time" in av_df.columns:
        if USE_NEGATIVE_TRAVEL_TIME_AS_REWARD:
            rewards = (-1.0 * av_df["travel_time"].astype(float)).tolist()
        else:
            rewards = av_df["travel_time"].astype(float).tolist()
    else:
        raise ValueError(
            f"{episode_csv} has neither 'reward' nor 'travel_time'.\n"
            f"Available columns are: {list(df.columns)}"
        )

    return rewards


def build_reward_table_from_episodes(
    output_csv=REWARD_CSV,
    episodes_folder=EPISODES_FOLDER,
    start_joint_action_id=0,
    end_joint_action_id=NB_JOINT_ACTIONS,
):
    """
    Builds a 2048-row payoff table:

        id,0,1,2,3,4,5,6,7,8,9

    There is no human 13 column because human 13 is not a coalition player.
    """
    rows = []

    for joint_action_id in range(start_joint_action_id, end_joint_action_id):
        episode_csv = find_episode_file(
            joint_action_id=joint_action_id,
            episodes_folder=episodes_folder,
        )

        av_rewards = episode_to_av_rewards(episode_csv)

        row = {"id": joint_action_id}

        for av_id, reward in zip(AV_IDS, av_rewards):
            row[str(av_id)] = reward

        rows.append(row)

        if joint_action_id % 100 == 0:
            print(f"Processed joint action {joint_action_id}/{end_joint_action_id - 1}")

    reward_df = pd.DataFrame(rows)
    reward_df.to_csv(output_csv, index=False)

    print(f"\nWrote payoff table to: {output_csv}")

    return reward_df


# ============================================================
# Encoding / decoding 2048 joint actions
# ============================================================

def int_to_strategy(idx, width):
    """
    Converts integer idx into a binary list of length width.
    """
    if idx < 0 or idx >= 2 ** width:
        raise ValueError(f"idx must be between 0 and {2 ** width - 1}")

    return [int(x) for x in format(idx, f"0{width}b")]


def joint_id_to_avs_and_human(joint_action_id, nb_avs=NB_AVS):
    """
    Decodes a 2048-row joint-action id.

    Encoding:
        first 10 bits -> AV actions
        last bit      -> human 13 action
    """
    bits = int_to_strategy(joint_action_id, nb_avs + 1)

    av_actions = bits[:nb_avs]
    human_action = bits[nb_avs]

    return av_actions, human_action


def avs_and_human_to_joint_id(av_actions, human_action):
    """
    Converts AV actions and human 13 action into the 2048-row joint-action id.
    """
    if len(av_actions) != NB_AVS:
        raise ValueError(f"Expected {NB_AVS} AV actions, got {len(av_actions)}")

    bits = list(av_actions) + [int(human_action)]

    if any(action not in [0, 1] for action in bits):
        raise ValueError("All actions must be binary: 0 or 1")

    return int("".join(map(str, bits)), 2)


def av_actions_to_club(av_actions):
    """
    Returns the AVs currently choosing action 1.
    """
    return [i for i, action in enumerate(av_actions) if action == 1]


# ============================================================
# Load payoff table
# ============================================================

def load_reward_table(filename=REWARD_CSV):
    """
    Loads the 2048-row payoff table.

    Required columns:
        id,0,1,2,3,4,5,6,7,8,9
    """
    df = pd.read_csv(filename)

    expected_columns = ["id"] + [str(i) for i in AV_IDS]
    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing columns in payoff table: {missing_columns}\n"
            f"Expected columns: {expected_columns}"
        )

    expected_rows = NB_JOINT_ACTIONS

    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows for 10 AVs + 1 human, "
            f"but found {len(df)} rows."
        )

    df = df.set_index("id").sort_index()

    return df


# ============================================================
# Payoff helpers
# ============================================================

def is_better(new_value, old_value):
    """
    Returns True if new_value is strictly better than old_value.
    """
    if HIGHER_IS_BETTER:
        return new_value > old_value

    return new_value < old_value


def payoff_gain(new_value, old_value):
    """
    Returns a gain value where positive means better.
    """
    if HIGHER_IS_BETTER:
        return new_value - old_value

    return old_value - new_value


def s_to_reward(av_actions, human_action, reward_df):
    """
    Returns AV payoffs for the given state.

    Output:
        {
            0: payoff_AV0,
            ...
            9: payoff_AV9
        }
    """
    joint_id = avs_and_human_to_joint_id(av_actions, human_action)
    row = reward_df.loc[joint_id]

    rewards = {}

    for av_id in AV_IDS:
        rewards[av_id] = float(row[str(av_id)])

    return rewards


def print_state(av_actions, human_action, reward_df):
    """
    Prints one state in a readable format.
    """
    joint_id = avs_and_human_to_joint_id(av_actions, human_action)
    club = av_actions_to_club(av_actions)
    rewards = s_to_reward(av_actions, human_action, reward_df)

    print("Joint action id:", joint_id)
    print("AV actions:", av_actions)
    print("AVs on action 1 / current club:", club)
    print("Human 13 action:", human_action)
    print("AV payoffs:", rewards)


# ============================================================
# Individual deviations: AVs only
# ============================================================

def individual_deviations_av_only(av_actions, human_action, reward_df):
    """
    Returns AVs that can improve by individually switching action.

    Human 13 is fixed and cannot deviate.
    """
    current_rewards = s_to_reward(av_actions, human_action, reward_df)

    deviators = []

    for av_id in AV_IDS:
        new_av_actions = av_actions.copy()
        new_av_actions[av_id] = 1 - new_av_actions[av_id]

        new_rewards = s_to_reward(new_av_actions, human_action, reward_df)

        if is_better(new_rewards[av_id], current_rewards[av_id]):
            deviators.append(av_id)

    return deviators


def nash_equilibrium_av_only(av_actions, human_action, reward_df):
    """
    Checks Nash equilibrium with respect to AV deviations only.

    Human 13 is fixed.
    """
    return len(individual_deviations_av_only(av_actions, human_action, reward_df)) == 0


# ============================================================
# Coalition deviations: AVs only
# ============================================================

def all_nonempty_av_coalitions():
    """
    Generates all non-empty AV coalitions.

    With 10 AVs, there are 2^10 - 1 = 1023 possible coalitions.
    """
    for size in range(1, NB_AVS + 1):
        for coalition in itertools.combinations(AV_IDS, size):
            yield list(coalition)


def coalition_can_deviate_av_only(coalition, av_actions, human_action, reward_df):
    """
    Checks whether a given AV coalition has any joint action change
    that makes every coalition member strictly better off.

    Human 13 is fixed and cannot be part of the coalition.

    Returns:
        can_deviate, new_av_actions, gains
    """
    if any(agent_id not in AV_IDS for agent_id in coalition):
        raise ValueError(
            f"Coalition {coalition} is invalid. "
            f"Only AVs {AV_IDS} can be in coalitions."
        )

    current_rewards = s_to_reward(av_actions, human_action, reward_df)
    current_coalition_actions = tuple(av_actions[i] for i in coalition)

    for new_actions_tuple in itertools.product([0, 1], repeat=len(coalition)):

        # Skip the case where the coalition changes nothing.
        if new_actions_tuple == current_coalition_actions:
            continue

        new_av_actions = av_actions.copy()

        for agent_id, new_action in zip(coalition, new_actions_tuple):
            new_av_actions[agent_id] = new_action

        new_rewards = s_to_reward(new_av_actions, human_action, reward_df)

        all_members_improve = all(
            is_better(new_rewards[i], current_rewards[i])
            for i in coalition
        )

        if all_members_improve:
            gains = {
                i: payoff_gain(new_rewards[i], current_rewards[i])
                for i in coalition
            }

            return True, new_av_actions, gains

    return False, None, None


def coalition_deviations_av_only(av_actions, human_action, reward_df):
    """
    Returns all AV coalitions that can profitably deviate from a state.

    Human 13 is fixed.
    """
    deviations = []

    for coalition in all_nonempty_av_coalitions():
        can_deviate, new_av_actions, gains = coalition_can_deviate_av_only(
            coalition=coalition,
            av_actions=av_actions,
            human_action=human_action,
            reward_df=reward_df,
        )

        if can_deviate:
            deviations.append(
                {
                    "coalition": coalition,
                    "new_av_actions": new_av_actions,
                    "gains": gains,
                }
            )

    return deviations


def strong_nash_equilibrium_av_only(av_actions, human_action, reward_df):
    """
    Checks strong Nash equilibrium with respect to AV coalitions only.

    Human 13 is fixed and cannot join coalitions.
    """
    return len(coalition_deviations_av_only(av_actions, human_action, reward_df)) == 0


# ============================================================
# Club-formation version: coalitions join action 1
# ============================================================

def clubs_that_can_form_from_state(av_actions, human_action, reward_df):
    """
    Finds AV clubs that can form by moving from action 0 to action 1.

    This matches the original club-formation idea:
        - current club = AVs already on action 1
        - new club members are AVs currently on action 0
        - a new club forms if all its members improve by switching to action 1
        - human 13 is fixed and cannot be in the club
    """
    current_rewards = s_to_reward(av_actions, human_action, reward_df)

    candidates = []

    outside_club = [av_id for av_id in AV_IDS if av_actions[av_id] == 0]

    for size in range(1, len(outside_club) + 1):
        for coalition_tuple in itertools.combinations(outside_club, size):
            coalition = list(coalition_tuple)

            new_av_actions = av_actions.copy()

            for av_id in coalition:
                new_av_actions[av_id] = 1

            new_rewards = s_to_reward(new_av_actions, human_action, reward_df)

            all_members_improve = all(
                is_better(new_rewards[i], current_rewards[i])
                for i in coalition
            )

            if all_members_improve:
                gains = {
                    i: payoff_gain(new_rewards[i], current_rewards[i])
                    for i in coalition
                }

                candidates.append(
                    {
                        "coalition": coalition,
                        "new_av_actions": new_av_actions,
                        "gains": gains,
                    }
                )

    return candidates


def explore_club_formation(initial_av_actions, human_action, reward_df, max_depth=10):
    """
    Recursively explores AV-only club formation.

    Human 13 is fixed.
    AVs can join action 1 if doing so improves every member of the forming club.
    """
    visited = set()

    def recurse(av_actions, depth):
        key = tuple(av_actions)

        if key in visited:
            return

        visited.add(key)

        indent = "    " * depth

        club = av_actions_to_club(av_actions)
        rewards = s_to_reward(av_actions, human_action, reward_df)

        ind_dev = individual_deviations_av_only(av_actions, human_action, reward_df)
        c_dev = coalition_deviations_av_only(av_actions, human_action, reward_df)

        is_nash = len(ind_dev) == 0
        is_strong = len(c_dev) == 0

        print(indent + "================================================")
        print(indent + f"Current AV club/action-1 set: {club}")
        print(indent + f"AV actions: {av_actions}")
        print(indent + f"Human 13 action fixed at: {human_action}")
        print(indent + f"AV payoffs: {rewards}")
        print(indent + f"AV individual deviations: {ind_dev}")
        print(indent + f"AV-only Nash equilibrium: {is_nash}")
        print(indent + f"AV-only strong Nash equilibrium: {is_strong}")

        if club:
            internally_stable = all(av_id not in ind_dev for av_id in club)
            print(indent + f"Club internally stable: {internally_stable}")

        forming_clubs = clubs_that_can_form_from_state(
            av_actions=av_actions,
            human_action=human_action,
            reward_df=reward_df,
        )

        print(indent + "New AV clubs that can form:")

        if not forming_clubs:
            print(indent + "    None")
        else:
            for item in forming_clubs:
                print(
                    indent
                    + f"    coalition {item['coalition']} "
                    + f"-> gains {item['gains']}"
                )

        if depth >= max_depth:
            print(indent + "Maximum recursion depth reached.")
            return

        for item in forming_clubs:
            recurse(item["new_av_actions"], depth + 1)

    recurse(initial_av_actions, depth=0)


# ============================================================
# Optional: summarize all 2048 states
# ============================================================

def summarize_all_states(reward_df, output_csv="coalition_summary_2048.csv"):
    """
    Checks all 2048 states and writes a summary CSV.

    The coalitions considered are AV-only.
    """
    rows = []

    for joint_action_id in range(NB_JOINT_ACTIONS):
        av_actions, human_action = joint_id_to_avs_and_human(joint_action_id)

        club = av_actions_to_club(av_actions)
        ind_dev = individual_deviations_av_only(av_actions, human_action, reward_df)
        c_dev = coalition_deviations_av_only(av_actions, human_action, reward_df)

        rows.append(
            {
                "joint_action_id": joint_action_id,
                "av_actions": "".join(map(str, av_actions)),
                "human13_action": human_action,
                "club": str(club),
                "individual_deviations_av_only": str(ind_dev),
                "num_coalition_deviations_av_only": len(c_dev),
                "nash_av_only": len(ind_dev) == 0,
                "strong_nash_av_only": len(c_dev) == 0,
            }
        )

        if joint_action_id % 100 == 0:
            print(f"Summarized state {joint_action_id}/{NB_JOINT_ACTIONS - 1}")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_csv, index=False)

    print(f"Wrote summary to: {output_csv}")

    return summary_df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    reward_path = Path(REWARD_CSV)

    if not reward_path.exists():
        print(f"{REWARD_CSV} was not found.")
        print("Building it from episode CSV files...")

        build_reward_table_from_episodes(
            output_csv=REWARD_CSV,
            episodes_folder=EPISODES_FOLDER,
            start_joint_action_id=0,
            end_joint_action_id=NB_JOINT_ACTIONS,
        )

    reward_df = load_reward_table(REWARD_CSV)

    initial_av_actions = [0 for _ in AV_IDS]

    # --------------------------------------------------------
    # Case 1: human 13 fixed to action 0
    # --------------------------------------------------------

    print("\n\n################################################")
    print("ANALYSIS WITH HUMAN 13 FIXED TO ACTION 0")
    print("################################################")

    human_action = 0

    print("\nInitial state:")
    print_state(initial_av_actions, human_action, reward_df)

    initial_clubs = clubs_that_can_form_from_state(
        av_actions=initial_av_actions,
        human_action=human_action,
        reward_df=reward_df,
    )

    print("\nInitial AV clubs that can form:")

    if not initial_clubs:
        print("    None")
    else:
        for item in initial_clubs:
            print(f"    Coalition {item['coalition']} -> gains {item['gains']}")

    print("\nRecursive club-formation tree:")

    explore_club_formation(
        initial_av_actions=initial_av_actions,
        human_action=human_action,
        reward_df=reward_df,
        max_depth=10,
    )

    # --------------------------------------------------------
    # Case 2: human 13 fixed to action 1
    # --------------------------------------------------------

    print("\n\n################################################")
    print("ANALYSIS WITH HUMAN 13 FIXED TO ACTION 1")
    print("################################################")

    human_action = 1

    print("\nInitial state:")
    print_state(initial_av_actions, human_action, reward_df)

    initial_clubs = clubs_that_can_form_from_state(
        av_actions=initial_av_actions,
        human_action=human_action,
        reward_df=reward_df,
    )

    print("\nInitial AV clubs that can form:")

    if not initial_clubs:
        print("    None")
    else:
        for item in initial_clubs:
            print(f"    Coalition {item['coalition']} -> gains {item['gains']}")

    print("\nRecursive club-formation tree:")

    explore_club_formation(
        initial_av_actions=initial_av_actions,
        human_action=human_action,
        reward_df=reward_df,
        max_depth=10,
    )

    # --------------------------------------------------------
    # Optional full 2048-state summary
    # --------------------------------------------------------

    print("\n\n################################################")
    print("WRITING FULL 2048-STATE SUMMARY")
    print("################################################")

    summarize_all_states(
        reward_df=reward_df,
        output_csv="coalition_summary_2048.csv",
    )