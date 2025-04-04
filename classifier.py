import os
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load .env variables (to keep the API key secure)
load_dotenv()
client = OpenAI(api_key=os.getenv("api_key"))

# Load the CSO Matrix typology from file
with open("data/cso-matrix.txt", "r", encoding="utf-8") as f:
    matrix_typology = f.read()

def classify_provision(provision_text):
    prompt = f"""
    Classify the following provision using the CSO Regulatory Regime Matrix. Use the following typology as your reference:
    {matrix_typology}
    Find the closest matching concept. If no exact match exists, choose the conceptually closest category.

    Classify the provision as either:
    - Restrictive: if it imposes barriers or burdens on CSO activity.
    - Permissive: if it enables, supports, or simplifies CSO activity.

    Assign the provision to one of the four CSO Matrix subgroups: Formation, Governance, Operations, Resources.
    Once assigned, do not change the category.

    Return a JSON object like this:

    {{
    "provision": "{provision_text}",
    "matched_matrix_provision": "Closest concept from matrix, as it appears in the matrix.",
    "subgroup": "Formation | Governance | Operations | Resources",
    "type": "Restrictive | Permissive",
    "explanation": "Brief legal reasoning and justification based on the matrix."
    }}
    """

    start_time = time.time()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a legal classification assistant trained in civil society regulation. Use the CSO Regulatory Matrix to classify provisions deterministically and return structured output. Do not create new categories or speculate beyond the matrix."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    usage = response.usage
    print("\nToken Usage:")
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
    print(f"Execution time: {duration} seconds\n")

    return response.choices[0].message.content