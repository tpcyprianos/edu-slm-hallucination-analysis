import pandas as pd

# ============================================================
# 1. Load the dataset
# ============================================================

df = pd.read_csv("socratic_method_interactions.csv")

file_output_name = "selected_samples_socratic_method.csv"

# ============================================================
# 2. Identify the model group
# ============================================================

def identify_model(row):
    """
    Identify the model group based on keywords
    found in model_version and model.
    """

    text = (
        str(row["model_version"]) + " " +
        str(row["model"])
    ).lower()

    if "llama" in text:
        return "llama"
    elif "qwen" in text:
        return "qwen"
    elif "gemma" in text:
        return "gemma"
    else:
        return None


df["model_group"] = df.apply(identify_model, axis=1)


# ============================================================
# 3. Sort the conversation turns
# ============================================================

df["turn"] = pd.to_numeric(df["turn"], errors="coerce")

df = df.sort_values(
    ["file_name", "conversation_name", "turn"]
).reset_index(drop=True)


# ============================================================
# 4. Associate each assistant response with the
#    preceding user message
# ============================================================

samples = []

for (file_name, conversation_name), conversation in df.groupby(
    ["file_name", "conversation_name"]
):

    conversation = conversation.sort_values("turn").reset_index(drop=True)

    for i, row in conversation.iterrows():

        # Keep only assistant messages
        if str(row["role"]).lower() != "assistant":
            continue

        # There must be a previous message
        if i == 0:
            continue

        previous_row = conversation.iloc[i - 1]

        # The previous message must be from the user
        if str(previous_row["role"]).lower() != "user":
            continue

        # Identify the model group
        model_group = row["model_group"]

        if model_group is None:
            continue

        # Store the assistant response together
        # with the preceding user message
        samples.append({
            "task": row["task"],
            "file_name": row["file_name"],
            "conversation_name": row["conversation_name"],
            "model_version": row["model_version"],
            "model_family": row["model_family"],
            "model_size": row["model_size"],
            "experiment": row["experiment"],
            "model": row["model"],
            "turn": row["turn"],

            # Previous user message
            "user_text": previous_row["text"],

            # Assistant response to be sampled
            "assistant_text": row["text"],

            "model_group": model_group
        })


samples_df = pd.DataFrame(samples)


# ============================================================
# 5. Check the number of available samples per model
# ============================================================

print("Available samples:")
print(samples_df["model_group"].value_counts())


# ============================================================
# 6. Randomly select 34 samples from each model
# ============================================================

selected = (
    samples_df
    .groupby("model_group", group_keys=False)
    .sample(
        n=34,
        random_state=42
    )
    .reset_index(drop=True)
)


# ============================================================
# 7. Check the final sample distribution
# ============================================================

print("\nSelected samples:")
print(selected["model_group"].value_counts())


# ============================================================
# 8. Save the selected samples
# ============================================================

selected.to_csv(
    file_output_name,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nFile saved as: {file_output_name}")