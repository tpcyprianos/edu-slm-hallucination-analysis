import os
# =========================
# SAVE RESULTS SAFELY
# =========================

def save_results(df, output_file, SEPARATOR):

    temp_file = output_file + ".tmp"

    df.to_csv(
        temp_file,
        sep=SEPARATOR,
        index=False,
        encoding="utf-8-sig"
    )

    os.replace(
        temp_file,
        output_file
    )