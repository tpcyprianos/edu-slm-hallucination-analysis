import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai
from report_generator import generate_html_report
from openai import OpenAI
from parse_evaluation import parse_evaluation

# =========================
# LOAD ENVIRONMENT VARIABLES
# =========================

load_dotenv()

# =========================
# INITIALIZE GEMINI
# =========================

#client = genai.Client(
#    api_key=os.getenv("GEMINI_API_KEY")
#)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#client = OpenAI(
#    base_url="http://localhost:1234/v1",
#    api_key="lm-studio"
#)

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
# SAVE RESULTS SAFELY
# =========================

def save_results(df, output_file):

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



# =========================
# MODEL CONFIGURATION
# =========================

MODEL = config["model"]
DELAY = config.get("delay", 13)
PROVIDER = config["provider"]

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
CRITERIA = config["criteria"]
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
print(f"Using model: {MODEL}")

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
# PROCESS DATA
# =========================

evaluation_columns = [
    "Hallucinate?",
    "Factual Fabrication",
    "Factual Contradiction",
    "Instruction Inconsistency",
    "Context Inconsistency",
    "Logical Inconsistency",
    "Comments"
]

# Create evaluation columns if they do not exist
for column in evaluation_columns:

    if column not in df.columns:

        df[column] = ""

    else:

        df[column] = df[column].astype(object)

# Create status column if it does not exist
if "Evaluation Status" not in df.columns:

    df["Evaluation Status"] = "pending"

for index, row in df.iterrows():

    # Evaluate only rows with the specified role
    if row[ROLE_COLUMN] != EVALUATION_ROLE:

        df.at[index, "Evaluation Status"] = "not_applicable"

        continue


    # Skip rows already evaluated
    if df.at[index, "Evaluation Status"] == "completed":

        print(
            f"Skipping row {index + 1}: "
            f"already evaluated."
        )

        continue


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

        {CRITERIA}

        {SCALE}

        Conversation context:

        Previous speaker:
        {context_role}

        Previous message:
        {context_text}

        Response to evaluate:

        {EVALUATION_ROLE}:
        {evaluation_text}

        Evaluate only the current response.

        Use the previous message exclusively as conversational context.

        IMPORTANT:
        Return ONLY a valid JSON object.
        Do not include Markdown, code fences, or any additional text.

        The JSON must contain exactly these fields:
        {json.dumps(OUTPUT_FORMAT, ensure_ascii=False, indent=2)}

        Use "Yes" or "No" for all classification fields.
        Use "Comments" for a brief explanation of your evaluation.
    """
    # =========================
    # CALL GEMINI
    # =========================

    #response = client.models.generate_content(
    #    model=MODEL,
    #    contents=full_prompt
    #)

    # =========================
    # CALL OPENAI
    # ========================= 
    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        )
    except Exception as error:

        print(
            f"Error evaluating row "
            f"{index + 1}: {error}"
        )

        df.at[index, "Evaluation Status"] = "error"

        df.at[index, "Comments"] = (
            f"Evaluation error: {error}"
        )

        save_results(
            df,
            OUTPUT_FILE
        )

        print(
            f"Error status saved for row "
            f"{index + 1}"
        )

        continue
    # =========================
    # CALL GEMMA
    # =========================

    #response = client.chat.completions.create(
    #    model=MODEL,
    #    messages=[
    #        {
    #            "role": "user",
    #            "content": full_prompt
    #        }
    #    ]
    #)

    # Store result - gemini
    #evaluation_results.append(
    #    response.text
    #)

    
    # Store result - gpt
    #response_text = response.choices[0].message.content

    #evaluation = parse_evaluation(
    #    response_text
    #)

    evaluation = parse_evaluation(
        response.choices[0].message.content
    )


    # Store evaluation directly in the DataFrame
    for column in evaluation_columns:

        value = evaluation.get(
            column,
            ""
        )

        if value is None:

            value = ""

        else:

            value = str(value)

        df.at[index, column] = value


    # Mark row as completed
    df.at[index, "Evaluation Status"] = "completed"

    # Save results immediately after evaluation
    save_results(
        df,
        OUTPUT_FILE
    )

    print(
        f"Evaluation saved for row "
        f"{index + 1}"
    )
        
    # Store result - gemma
    #results.append(response.choices[0].message.content)
    #evaluation = parse_evaluation(
    #    response.choices[0].message.content
    #)
    #evaluation_results.append(evaluation)

    # =========================
    # RATE LIMIT CONTROL
    # =========================

    if index < len(df) - 1:

        print(
            f"Waiting {DELAY} seconds..."
        )

        time.sleep(DELAY)

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
    group_columns=GROUP_COLUMNS,
    model = MODEL
)

print("Generating HTML report...")

# =========================
# FINISH
# =========================

print("\nEvaluation completed!")