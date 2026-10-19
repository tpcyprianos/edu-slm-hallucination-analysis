import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score


# ============================================================
# CONFIGURATION
# ============================================================

FILES = {
    "Judge_1": "results_examples_anthropic.csv",
    "Judge_2": "results_examples_gemini.csv",
    "Judge_3": "results_examples_gpt.csv"
}

RATING_COLUMNS = [
    "Hallucinate?",
    "Factual Fabrication",
    "Factual Contradiction",
    "Instruction Inconsistency",
    "Context Inconsistency",
    "Logical Inconsistency"
]

# Columns that identify the same evaluated interaction
KEY_COLUMNS = [
    "task",
    "file_name",
    "conversation_name",
    "experiment",
    "model",
    "turn"
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_rating(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    # Positive
    if value in ["1", "yes", "y", "true"]:
        return 1

    # Negative
    if value in ["0", "no", "n", "false"]:
        return 0

    # Missing / not applicable
    if value in [
        "",
        "na",
        "n/a",
        "not applicable",
        "not_applicable",
        "none",
        "nan"
    ]:
        return np.nan

    # Unknown value
    return np.nan


# ============================================================
# LOAD DATA
# ============================================================

judges = {}

for judge_name, filename in FILES.items():

    df = pd.read_csv(filename)

    print(f"{judge_name}: {len(df)} rows loaded")

    # Clean Evaluation Status
    df["Evaluation Status"] = (
        df["Evaluation Status"]
        .astype(str)
        .str.replace("<br>", "", regex=False)
        .str.strip()
        .str.lower()
    )

    # Keep only completed evaluations
    df = df[
        df["Evaluation Status"] == "completed"
    ].copy()

    # Normalize ratings
    for col in RATING_COLUMNS:
        df[col] = df[col].apply(normalize_rating)

    judges[judge_name] = df

    print(f"{judge_name}: {len(df)} completed rows")


# ============================================================
# PAIRWISE COHEN'S KAPPA
# ============================================================

results = []

judge_names = list(judges.keys())


for i in range(len(judge_names)):

    for j in range(i + 1, len(judge_names)):

        judge_a = judge_names[i]
        judge_b = judge_names[j]

        df_a = judges[judge_a]
        df_b = judges[judge_b]

        print(
            f"\nComparing {judge_a} vs {judge_b}"
        )

        # Keep only identifying columns + ratings
        df_a = df_a[
            KEY_COLUMNS + RATING_COLUMNS
        ].copy()

        df_b = df_b[
            KEY_COLUMNS + RATING_COLUMNS
        ].copy()

        # Merge the two judges
        merged = pd.merge(
            df_a,
            df_b,
            on=KEY_COLUMNS,
            how="inner",
            suffixes=("_A", "_B")
        )

        print(
            f"Paired observations: {len(merged)}"
        )

        # ----------------------------------------------------
        # Calculate kappa for each evaluated model
        # ----------------------------------------------------

        for model in sorted(
            merged["model"].dropna().unique()
        ):

            model_data = merged[
                merged["model"] == model
            ].copy()

            for dimension in RATING_COLUMNS:

                col_a = f"{dimension}_A"
                col_b = f"{dimension}_B"

                data = model_data[
                    [col_a, col_b]
                ].dropna()

                n = len(data)

                if n == 0:
                    continue

                kappa = cohen_kappa_score(
                    data[col_a],
                    data[col_b]
                )

                agreement = (
                    data[col_a] == data[col_b]
                ).mean()

                results.append({
                    "Judge_A": judge_a,
                    "Judge_B": judge_b,
                    "Model": model,
                    "Dimension": dimension,
                    "N": n,
                    "Agreement": agreement,
                    "Cohen_kappa": kappa
                })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n================ RESULTS ================\n")

print(
    results_df.to_string(index=False)
)

# Save results
results_df.to_csv(
    "cohen_kappa_results.csv",
    index=False
)

print(
    "\nResults saved to cohen_kappa_results.csv"
)


print(results_df)

print("\n" + "=" * 80)
print("COHEN'S KAPPA RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

summary = results_df.pivot_table(
    index="Dimension",
    columns="Model",
    values="Cohen_kappa"
)

print("\n")
print("=" * 80)
print("COHEN'S KAPPA BY DIMENSION AND MODEL")
print("=" * 80)

print(
    summary.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)