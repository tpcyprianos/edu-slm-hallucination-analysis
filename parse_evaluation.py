import os
import time
import json
import pandas as pd

def parse_evaluation(response_text):

    try:
        evaluation = json.loads(response_text)

        return {
            "Hallucinate?": evaluation.get("Hallucinate?", ""),
            "Factual Fabrication": evaluation.get(
                "Factual Fabrication", ""
            ),
            "Factual Contradiction": evaluation.get(
                "Factual Contradiction", ""
            ),
            "Instruction Inconsistency": evaluation.get(
                "Instruction Inconsistency", ""
            ),
            "Context Inconsistency": evaluation.get(
                "Context Inconsistency", ""
            ),
            "Logical Inconsistency": evaluation.get(
                "Logical Inconsistency", ""
            ),
            "Comments": evaluation.get("Comments", "")
        }

    except json.JSONDecodeError:

        print("Warning: Model returned invalid JSON.")
        print(response_text)

        return {
            "Hallucinate?": "",
            "Factual Fabrication": "",
            "Factual Contradiction": "",
            "Instruction Inconsistency": "",
            "Context Inconsistency": "",
            "Logical Inconsistency": "",
            "Comments": response_text
        }