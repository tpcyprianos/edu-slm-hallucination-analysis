import os
from html import escape


def generate_html_report(
    df,
    html_file,
    text_column,
    role_column,
    turn_column,
    evaluation_role,
    group_columns
):

    html_parts = []

    # =========================
    # HTML HEADER AND STYLE
    # =========================

    html_parts.append("""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>LLM Evaluation Report</title>

<style>

body {
    font-family: Arial, sans-serif;
    background-color: #f5f7fb;
    margin: 0;
    padding: 40px;
    color: #1f2937;
}

.container {
    max-width: 1100px;
    margin: auto;
}

h1 {
    margin-bottom: 10px;
}

.subtitle {
    color: #6b7280;
    margin-bottom: 40px;
}

.summary {
    background: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 30px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.card {
    background: white;
    border-radius: 12px;
    padding: 25px;
    margin-bottom: 25px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}

.metadata {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}

.tag {
    background: #eef2ff;
    color: #4338ca;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 14px;
}

.section-title {
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 8px;
}

.message {
    background: #f9fafb;
    padding: 15px;
    border-radius: 6px;
    white-space: pre-wrap;
    line-height: 1.6;
}

.context-message {
    border-left: 4px solid #3b82f6;
}

.assistant-message {
    border-left: 4px solid #10b981;
}

.evaluation {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    padding: 18px;
    border-radius: 6px;
    white-space: pre-wrap;
    line-height: 1.6;
}

</style>

</head>

<body>

<div class="container">

<h1>LLM Evaluation Report</h1>

<p class="subtitle">
Automatically generated evaluation report
</p>
""")

    # =========================
    # SUMMARY
    # =========================

    evaluated_rows = df[
        df["evaluation"].notna()
        & (df["evaluation"] != "")
    ]

    html_parts.append(f"""
<div class="summary">

<strong>Total rows:</strong> {len(df)}<br>

<strong>Evaluated responses:</strong>
{len(evaluated_rows)}

</div>
""")

    # =========================
    # PROCESS EVALUATED ROWS
    # =========================

    for index, row in df.iterrows():

        # Only display evaluated rows
        if row[role_column] != evaluation_role:
            continue

        evaluation = row.get("evaluation", "")

        if not evaluation:
            continue

        # =========================
        # GET PREVIOUS ROW
        # =========================

        previous_text = "No previous context available."
        previous_role = ""

        if index > 0:

            previous_row = df.iloc[index - 1]

            same_conversation = all(
                previous_row[column] == row[column]
                for column in group_columns
            )

            if same_conversation:

                previous_text = previous_row[text_column]
                previous_role = previous_row[role_column]

        # =========================
        # ESCAPE HTML CONTENT
        # =========================

        previous_text = escape(str(previous_text))
        current_text = escape(str(row[text_column]))
        evaluation_text = escape(str(evaluation))

        # =========================
        # BUILD METADATA
        # =========================

        metadata = ""

        for column in group_columns:

            metadata += f"""
<span class="tag">
{escape(column)}: {escape(str(row[column]))}
</span>
"""

        metadata += f"""
<span class="tag">
Turn: {escape(str(row[turn_column]))}
</span>
"""

        # =========================
        # ADD EVALUATION CARD
        # =========================

        html_parts.append(f"""

<div class="card">

<div class="metadata">
{metadata}
</div>

<div class="section-title">
Conversation context ({escape(str(previous_role))})
</div>

<div class="message context-message">
{previous_text}
</div>

<div class="section-title">
Response evaluated ({escape(str(row[role_column]))})
</div>

<div class="message assistant-message">
{current_text}
</div>

<div class="section-title">
Evaluation
</div>

<div class="evaluation">
{evaluation_text}
</div>

</div>

""")

    # =========================
    # CLOSE HTML
    # =========================

    html_parts.append("""

</div>

</body>
</html>
""")

    # =========================
    # CREATE OUTPUT DIRECTORY
    # =========================

    html_directory = os.path.dirname(html_file)

    if html_directory:

        os.makedirs(
            html_directory,
            exist_ok=True
        )

    # =========================
    # SAVE HTML FILE
    # =========================

    with open(
        html_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(html_parts)
        )

    print(f"HTML report saved to: {html_file}")