import pandas as pd
import numpy as np
import pingouin as pg

def normalize_rating(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    # Valores positivos
    if value in ["1", "yes", "y", "true"]:
        return 1

    # Valores negativos
    if value in ["0", "no", "n", "false"]:
        return 0

    # Valores que não representam uma avaliação
    if value in [
        "",
        "not_applicable",
        "not applicable",
        "n/a",
        "na",
        "nan",
        "none"
    ]:
        return np.nan

    # Qualquer outro valor é considerado não reconhecido
    return np.nan

judge1 = pd.read_csv("results/results_examples_anthropic.csv")
judge2 = pd.read_csv("results/results_examples_gpt.csv")


for df in [judge1, judge2]:

    df["Evaluation Status"] = (
        df["Evaluation Status"]
        .astype(str)
        .str.replace("<br>", "", regex=False)
        .str.strip()
        .str.lower()
    )

print(judge1.shape)
print(judge2.shape)

rating_columns = [
    "Hallucinate?",
    "Factual Fabrication",
    "Factual Contradiction",
    "Instruction Inconsistency",
    "Context Inconsistency",
    "Logical Inconsistency"
]

for col in rating_columns:
    print("\n", col)
    print("Judge 1:")
    print(judge1[col].value_counts(dropna=False))

    print("Judge 2:")
    print(judge2[col].value_counts(dropna=False))


judge1 = judge1[
    judge1["Evaluation Status"] == "completed"
].copy()

judge2 = judge2[
    judge2["Evaluation Status"] == "completed"
].copy()

for col in rating_columns:

    judge1[col] = judge1[col].apply(normalize_rating)
    judge2[col] = judge2[col].apply(normalize_rating)

for col in rating_columns:
    print("\n", col)
    print("Judge 1:", judge1[col].value_counts(dropna=False).to_dict())
    print("Judge 2:", judge2[col].value_counts(dropna=False).to_dict())

key_columns = [
    "task",
    "file_name",
    "turn"
]

print(
    judge1.duplicated(key_columns).sum()
)

print(
    judge2.duplicated(key_columns).sum()
)

### Pareamento dos juizes

merged = pd.merge(
    judge1[
        key_columns + rating_columns
    ],
    judge2[
        key_columns + rating_columns
    ],
    on=key_columns,
    how="inner",
    suffixes=("_judge1", "_judge2")
)

print("Judge 1:", len(judge1))
print("Judge 2:", len(judge2))
print("Paired:", len(merged))

def calculate_icc(merged, rating):

    col_judge1 = f"{rating}_judge1"
    col_judge2 = f"{rating}_judge2"

    data = merged[
        key_columns + [col_judge1, col_judge2]
    ].copy()

    # Remover itens nos quais algum dos juízes não forneceu avaliação
    data = data.dropna(
        subset=[col_judge1, col_judge2]
    )

    # Criar identificador único do item
    data["item"] = (
        data["task"].astype(str)
        + "_"
        + data["file_name"].astype(str)
        + "_turn_"
        + data["turn"].astype(str)
    )

    # Formato long
    long_data = pd.concat([
        data[["item", col_judge1]].rename(
            columns={col_judge1: "Score"}
        ).assign(Judge="Judge_1"),

        data[["item", col_judge2]].rename(
            columns={col_judge2: "Score"}
        ).assign(Judge="Judge_2")
    ])

    # ICC
    icc = pg.intraclass_corr(
        data=long_data,
        targets="item",
        raters="Judge",
        ratings="Score"
    )

    return icc, long_data

def calculate_icc(merged, rating):

    col_judge1 = f"{rating}_judge1"
    col_judge2 = f"{rating}_judge2"

    data = merged[
        key_columns + [col_judge1, col_judge2]
    ].copy()

    # Remover itens nos quais algum dos juízes não forneceu avaliação
    data = data.dropna(
        subset=[col_judge1, col_judge2]
    )

    # Criar identificador único do item
    data["item"] = (
        data["task"].astype(str)
        + "_"
        + data["file_name"].astype(str)
        + "_turn_"
        + data["turn"].astype(str)
    )

    # Formato long
    long_data = pd.concat([
        data[["item", col_judge1]].rename(
            columns={col_judge1: "Score"}
        ).assign(Judge="Judge_1"),

        data[["item", col_judge2]].rename(
            columns={col_judge2: "Score"}
        ).assign(Judge="Judge_2")
    ])

    # ICC
    icc = pg.intraclass_corr(
        data=long_data,
        targets="item",
        raters="Judge",
        ratings="Score"
    )

    return icc, long_data

icc_results = {}

for rating in rating_columns:

    icc, data = calculate_icc(
        merged,
        rating
    )

    icc_results[rating] = icc

    results = []

for rating in rating_columns:

    icc, data = calculate_icc(
        merged,
        rating
    )

    row = icc[
        icc["Type"] == "ICC2"
    ].iloc[0]

    results.append({
        "Variable": rating,
        "N": data["item"].nunique(),
        "ICC": row["ICC"],
        "CI95%": row["CI95%"],
        "F": row["F"],
        "p": row["pval"]
    })

results_df = pd.DataFrame(results)

print(results_df)