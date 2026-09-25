import json
import re


def parse_evaluation(response):

    try:

        # Try to parse the entire response as JSON
        return json.loads(response)

    except json.JSONDecodeError:

        # Search for the first JSON object in the response
        match = re.search(
            r"\{.*\}",
            response,
            re.DOTALL
        )

        if match:

            json_text = match.group(0)

            try:
                evaluation = json.loads(json_text)
                #Normalization spaces
                evaluation = {
                    re.sub(r"\s+", " ", key).strip(): value
                    for key, value in evaluation.items()
                            }
                return evaluation

            except json.JSONDecodeError:

                print(
                    "Warning: JSON object found, "
                    "but it could not be parsed."
                )

                print(
                    "Extracted JSON:"
                )

                print(json_text)

                return None

        print(
            "Warning: Model returned invalid JSON."
        )

        print(
            "Raw model response:"
        )

        print(response)

        return None