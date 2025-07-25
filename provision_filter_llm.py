import os
import time
import datetime
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import json

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("api_key"))

# Load extracted provisions
df = pd.read_csv("outputs/provisions_from_spacing_france.csv")
df["label"] = ""
df["explanation"] = ""

# Define structured output schema
schema = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["provision", "not_provision"]
        },
        "explanation": {
            "type": "string"
        }
    },
    "required": ["label", "explanation"],
    "additionalProperties": False
}

# Instructions for GPT
instructions = """
    You are a legal assistant tasked with identifying whether a paragraph from a legal document is a standalone legal provision.
    You will receive fragments of text from a legal document, and your job is to classify each paragraph based on its content, whether it regulates legal matters or not.

    Classify each paragraph as:
    "provision": if it establishes a legal rule, right, obligation, restriction, or process. It must contain enforceable or actionable content
    "not_provision": if it is a title, part heading, metadata, citation, date, or reference to other sections.

    Here are examples:
    1. 'PART III - GENERAL PROVISIONS': not_provision
    2. 'Section 3. A person shall not participate in a public assembly without notifying authorities.': provision
    3. '[Date of assent: June 13, 1950]': not_provision

    Respond using the structured format.
"""

# Classify each paragraph
for i, row in df.iterrows():
    text = row["text"]
    print(f"Classifying paragraph {i}...")

    try:
        response = client.responses.create(
            model="gpt-4.1",
            instructions=instructions,
            input=text,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "provision_filter",
                    "strict": True,
                    "schema": schema
                }
            },
            temperature=0.2
        )
        print(f"Raw output at row {i}:\n{response.output}\n")

        
        # Safe parsing
        if response.output and response.output[0].content:
            raw_text = response.output[0].content[0].text
            parsed = json.loads(raw_text)

            df.at[i, "label"] = parsed.get("label", "")
            df.at[i, "explanation"] = parsed.get("explanation", "")
        else:
            raise ValueError("Empty response content")
        
    except Exception as e:
        print(f"Error at row {i}: {e}")
        df.at[i, "label"] = ""
        df.at[i, "explanation"] = f"FAILED: {e}"

    time.sleep(1.2)  # Respect rate limits

# Save results to timestamped CSV
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"outputs/filtered_provisions_{timestamp}.csv"
df.to_csv(output_path, index=False)
print(f"\n Saved filtered provisions to: {output_path}")

# Save retry list for failed rows
failed = df[df["label"] == ""]
if not failed.empty:
    retry_path = f"outputs/retry_failed_provisions_{timestamp}.csv"
    failed.to_csv(retry_path, index=False)
    print(f" Saved retry file for {len(failed)} failed rows: {retry_path}")