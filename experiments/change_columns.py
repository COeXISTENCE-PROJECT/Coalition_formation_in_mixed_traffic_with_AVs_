import pandas as pd

input_file = "reward_df_9_31_5_15agents.csv"
output_file = "reward_df_9_31_5_15agents.csv"

df = pd.read_csv(input_file)

# Save original column titles
columns = df.columns.tolist()

# Swap the DATA in the 3rd and 4th columns, but keep column titles unchanged
df.iloc[:, [4, 5]] = df.iloc[:, [5, 4]].values

# Restore original column titles
df.columns = columns

df.to_csv(output_file, index=False)

print(f"Saved swapped file as: {output_file}")