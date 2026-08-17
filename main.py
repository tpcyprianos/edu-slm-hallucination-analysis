import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai
from report_generator import generate_html_report

# =========================
# LOAD ENVIRONMENT VARIABLES
# =========================

load_dotenv()

# =========================
# INITIALIZE GEMINI
# =========================

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# =========================
# LOAD CONFIGURATION
# =========================

with open("config.json", "r", encoding="utf-8") as file:
    config = json.load(file)

# =========================
# FILE CONFIGURATION
# =========================

INPUT_FILE = config["input_file"]
OUTPUT_FILE = config["output_file"]
SEPARATOR = config.get("separator", ",")
HTML_OUTPUT_FILE = config.get(
    "html_output_file",
    "results/report.html"
)

# =========================
# MODEL CONFIGURATION
# =========================

MODEL = config["model"]
DELAY = config.get("delay", 13)

# =========================
# DATA STRUCTURE CONFIGURATION
# =========================

TEXT_COLUMN = config.get("text_column", "text")
ROLE_COLUMN = config.get("role_column", "role")
TURN_COLUMN = config.get("turn_column", "turn")


EVALUATION_ROLE = config.get(
    "evaluation_role",
    "assistant"
)

GROUP_COLUMNS = config.get(
    "group_columns",
    ["task", "conversation_name"]
)

# =========================
# PROMPT CONFIGURATION
# =========================

ROLE = config["role"]
DOMAIN = config["domain_reference"]
RULES = config["tutor_rules_under_evaluation"]
CRITERIA = config["criteria"]
NOT_HAL = config["not_hallucinations"]
PRECEDENCE = config["precedence"]
PROCEDURE = config["procedure"]
EVIDENCE = config["evidence_bar"]
SCALE = config["scale"]
OUTPUT_FORMAT = config["output_format"]


# =========================
# READ DATA
# =========================

df = pd.read_csv(
    INPUT_FILE,
    sep=SEPARATOR
)

# =========================
# SORT DATA
# =========================

sort_columns = GROUP_COLUMNS + [TURN_COLUMN]

df = df.sort_values(
    by=sort_columns
).reset_index(drop=True)


print(f"Total number of rows: {len(df)}")

# =========================
# VALIDATE COLUMNS
# =========================

required_columns = (
    GROUP_COLUMNS
    + [
        TEXT_COLUMN,
        ROLE_COLUMN,
        TURN_COLUMN
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in the input file: {missing_columns}"
    )

# =========================
# PROCESS DATA
# =========================

results = []

for index, row in df.iterrows():

    # Evaluate only rows with the specified role
    if row[ROLE_COLUMN] != EVALUATION_ROLE:

        results.append("")
        continue


    print(
        f"Evaluating row "
        f"{index + 1}/{len(df)}..."
    )


    # =========================
    # CURRENT TEXT TO EVALUATE
    # =========================

    evaluation_text = row[TEXT_COLUMN]


    # =========================
    # GET PREVIOUS ROW AS CONTEXT
    # =========================

    context_text = "No previous context available."
    context_role = ""

    if index > 0:

        previous_row = df.iloc[index - 1]

        # Check whether the previous row
        # belongs to the same conversation

        same_conversation = all(
            previous_row[column] == row[column]
            for column in GROUP_COLUMNS
        )

        if same_conversation:

            context_text = previous_row[TEXT_COLUMN]

            context_role = previous_row[ROLE_COLUMN]


    # =========================
    # BUILD PROMPT
    # =========================

    full_prompt = f"""
    {ROLE}

    {DOMAIN}

    {RULES}

    {CRITERIA}

    {NOT_HAL}

    {PRECEDENCE}

    {PROCEDURE}

    {EVIDENCE}

    {SCALE}

    {OUTPUT_FORMAT}    
    """


    # =========================
    # CALL GEMINI
    # =========================

    response = client.models.generate_content(
        model=MODEL,
        contents=full_prompt
    )


    # Store result

    results.append(
        response.text
    )


    # =========================
    # RATE LIMIT CONTROL
    # =========================

    if index < len(df) - 1:

        print(
            f"Waiting {DELAY} seconds..."
        )

        time.sleep(DELAY)


# =========================
# ADD RESULTS
# =========================

df["evaluation"] = results


# =========================
# CREATE OUTPUT DIRECTORY
# =========================

output_directory = os.path.dirname(
    OUTPUT_FILE
)

if output_directory:

    os.makedirs(
        output_directory,
        exist_ok=True
    )


# =========================
# SAVE RESULTS
# =========================

df.to_csv(
    OUTPUT_FILE,
    sep=SEPARATOR,
    index=False,
    encoding="utf-8-sig"
)


print(
    f"Results saved to: {OUTPUT_FILE}"
)

# =========================
# GENERATE HTML REPORT
# =========================

generate_html_report(
    df=df,
    html_file=HTML_OUTPUT_FILE,
    text_column=TEXT_COLUMN,
    role_column=ROLE_COLUMN,
    turn_column=TURN_COLUMN,
    evaluation_role=EVALUATION_ROLE,
    group_columns=GROUP_COLUMNS
)

print("Generating HTML report...")

# =========================
# FINISH
# =========================

print("\nEvaluation completed!")