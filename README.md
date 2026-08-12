# Hallucination Analysis of Small Language Models

## Configuration

The `config.json` file controls the main settings of the evaluation process, including input and output files, model parameters, dataset structure, conversation grouping, and evaluation instructions.

Example:

```json
{
    "input_file": "examples_interactions.csv",
    "output_file": "results/results_examples.csv",
    "separator": ";",
    "html_output_file": "results/report.html",

    "model": "gemini-3.5-flash",
    "delay": 13,

    "text_column": "text",
    "role_column": "role",
    "turn_column": "turn",

    "evaluation_role": "assistant",

    "group_columns": [
        "task",
        "conversation_name"
    ],

    "role": "You are an evaluator of language model responses.",

    "criteria": "Evaluate each language model response according to the following criteria...",

    "scale": "Evaluate each criterion using the defined evaluation scale..."
}
```

### File Configuration

| Parameter | Description |
|---|---|
| `input_file` | Path to the input CSV file containing the interaction data to be evaluated. |
| `output_file` | Path where the CSV file containing the original data and the generated evaluations will be saved. |
| `separator` | Delimiter used in the CSV file, such as `;` or `,`. |
| `html_output_file` | Path where the HTML report containing the evaluated interactions will be generated. |

Example:

```json
"input_file": "examples_interactions.csv",
"output_file": "results/results_examples.csv",
"separator": ";",
"html_output_file": "results/report.html"
```

---

### Model Configuration

| Parameter | Description |
|---|---|
| `model` | The language model used to perform the evaluations. |
| `delay` | Time, in seconds, between consecutive API requests. This can help prevent rate-limit errors. |

Example:

```json
"model": "gemini-3.5-flash",
"delay": 13
```

The appropriate value for `delay` may depend on the API quota and rate limits associated with the selected model.

---

### Dataset Structure

The following parameters define which columns in the input dataset contain the relevant information:

| Parameter | Description |
|---|---|
| `text_column` | Name of the column containing the textual content of each conversation turn. |
| `role_column` | Name of the column identifying the speaker or role associated with each turn, such as `user` or `assistant`. |
| `turn_column` | Name of the column indicating the order of the turns within a conversation. |

Example:

```json
"text_column": "text",
"role_column": "role",
"turn_column": "turn"
```

For instance, the input dataset may have the following structure:

```text
task;file_name;conversation_name;model_version;model_family;model_size;experiment;model;turn;role;text
Task 3.10;3.10_Llama_Examples.json;Understanding Digit Placement;310;Llama;;Examples;llama-3.2-1b-instruct.gguf;1;user;System: You are a helpful math tutor[...].
Task 3.10;3.10_Llama_Examples.json;Understanding Digit Placement;310;Llama;;Examples;llama-3.2-1b-instruct.gguf;2;assistant;"Let's break it down step by step[...].
```

---

### Evaluation Target

The `evaluation_role` parameter specifies which conversation turns should be evaluated.

```json
"evaluation_role": "assistant"
```

With this configuration, only rows in which the `role` column contains `assistant` will be evaluated. The remaining rows are preserved in the output dataset but will not receive an evaluation.

---

### Conversation Grouping

The `group_columns` parameter defines which columns identify a conversation.

```json
"group_columns": [
    "task",
    "conversation_name"
]
```

These columns are used to ensure that contextual information is retrieved only from the same conversation.

Before evaluating a response, the system checks whether the previous row belongs to the same group defined by `group_columns`. This prevents contextual information from one conversation or task from being incorrectly used to evaluate a response from another conversation.

The grouping structure can be adapted to different datasets. For example:

```json
"group_columns": [
    "participant_id",
    "session_id"
]
```

---

### Evaluation Instructions

The following parameters define the instructions provided to the language model.

#### `role`

Defines the role or perspective that the language model should adopt during the evaluation.

Example:

```json
"role": "You are an evaluator of language model responses."
```

#### `criteria`

Defines the criteria that the model should use to evaluate each response.

Example:

```json
"criteria": "Evaluate each language model response according to the following criteria: correctness, relevance, clarity, and pedagogical appropriateness."
```

#### `scale`

Defines how the evaluation criteria should be assessed or scored.

Example:

```json
"scale": "Evaluate each criterion using a scale from 1 to 5, where 1 represents very poor performance and 5 represents excellent performance."
```

---

### Adapting the Configuration

The `config.json` file can be modified to support different datasets and evaluation protocols without requiring changes to the main Python code.

For example, users can change:

- input and output file paths;
- the CSV separator;
- the language model;
- the delay between API requests;
- dataset column names;
- the role to be evaluated;
- the columns used to identify conversations;
- the evaluation criteria;
- the evaluation scale.

This design allows the evaluation pipeline to be reused across different datasets and evaluation scenarios while keeping the core implementation unchanged.