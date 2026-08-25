import pandas as pd

# ============================================================
# 1. Ler o dataset
# ============================================================

df = pd.read_csv("examples_interactions.csv")


# ============================================================
# 2. Identificar o grupo do modelo
# ============================================================

def identify_model(row):

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
# 3. Ordenar os turnos
# ============================================================

df["turn"] = pd.to_numeric(df["turn"], errors="coerce")

df = df.sort_values(
    ["file_name", "conversation_name", "turn"]
).reset_index(drop=True)


# ============================================================
# 4. Associar USER + ASSISTANT
# ============================================================

samples = []

for (file_name, conversation_name), conversation in df.groupby(
    ["file_name", "conversation_name"]
):

    conversation = conversation.sort_values("turn").reset_index(drop=True)

    for i, row in conversation.iterrows():

        # Queremos somente assistant
        if str(row["role"]).lower() != "assistant":
            continue

        # Precisa existir uma mensagem anterior
        if i == 0:
            continue

        previous_row = conversation.iloc[i - 1]

        # A mensagem anterior precisa ser user
        if str(previous_row["role"]).lower() != "user":
            continue

        # Identificar modelo
        model_group = row["model_group"]

        if model_group is None:
            continue

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

            "user_text": previous_row["text"],
            "assistant_text": row["text"],

            "model_group": model_group
        })


samples_df = pd.DataFrame(samples)


# ============================================================
# 5. Verificar quantidade disponível
# ============================================================

print("Amostras disponíveis:")
print(samples_df["model_group"].value_counts())


# ============================================================
# 6. Selecionar 34 aleatoriamente de cada modelo
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
# 7. Verificar resultado
# ============================================================

print("\nAmostras selecionadas:")
print(selected["model_group"].value_counts())


# ============================================================
# 8. Salvar novo CSV
# ============================================================

selected.to_csv(
    "selected_samples.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivo salvo como: selected_samples.csv")