import os
from openai import OpenAI
import time
import json

if os.getenv("HF_SPACE") != "true":
    from dotenv import load_dotenv
    load_dotenv()

client = OpenAI(api_key=os.getenv("api_key"))

# Load the CSO Matrix typology from file
with open("data/cso-matrix.txt", "r", encoding="utf-8") as f:
    matrix_typology = f.read()

def code_law(law_text):
    prompt = f"""
    Code the following law: {law_text}
    """

    system_instructions = f"""
        You are a legal analyst trained to analyze civil society regulation. You will help code a legal regime by checking for the presence, absence, or negation of standard regulatory provisions.
        TASK:
        1. Analyze the legal statute to understand its objective and what it regulates.
        2. For EACH of the 56 standardized provisions in the CSO Matrix (listed below):
        - Code as 1 if one or more sections of the law EXPLICITLY IMPLEMENT the rule.
        - Code as -1 if one or more sections of the law EXPLICITLY NEGATE or forbid the rule.
        - Code as 0 if the law is silent on the matter or does not clearly address the rule.

        If partially present or ambiguous, explain your reasoning in a short note.

        Important: The law may use a different terminology, but the matrix uses CSO (Civil Society Organization) as an umbrella term that includes terms such as charity, foundation, non-profit organizations and other kinds of voluntary associations.

        Before coding the law, provide:
        Name: Name of law (if available)
        Jurisdiction: Where the law applies
        Year: Year of enactment (if available)
        Language: The language the provided statute is in
        Objective: What the statute is intended to regulate, and if it's related to CSO activity

        Then, for each matrix provision, provide:

        Output format:
        Provision Number/ID
        Text of matrix provision (exactly as it appears in matrix)
        Code: 1, 0, or -1
        Brief explanation (1-2 sentences, only if ambiguous or partial)

        Example Output: All in a JSON file with the following structure:

        {{
        "Name": "Ley 39",
        "Jurisdiction": "Panama",
        "Year": 2018,
        "Language": "Spanish",
        "Objective": "The law is intended to regulate the creation of “public interest associations”, which directly affects the formation and thus activity of CSOs.",
        "Provisions": [
            {{
            "Provision": "1. PERMISSIVE RESOURCES 1 [CSOs] [must] [report their finances for public access] [after law's commencement] [or else face penalty for non-compliance]",
            "Code": 1,
            "Explanation": "The Public Benefit Organizations Act, Section X, requires all CSOs to publish books of accounts for the public annually."
            }},
            {{
            "Provision": "2. RESTRICTIVE OPERATIONS 3. [CSOs] [must not] [exceed specific threshold of budget spent on overhead] such as a certain percentage of budgets spent on administrative costs [after law's commencement] [or else face penalty for non-compliance]",
            "Code": 0,
            "Explanation": "No provision found establishing a threshold of overhead spending."
            }},
            {{
            "Provision": "3. PERMISSIVE FORMATION 3. [Agency] [must not] [reject registration for reasons other than those explicitly stated] [after law's commencement] [or else it is overstepping its authority]",
            "Code": -1,
            "Explanation": "Section Y, Article 8 states that the government reserves the right to reject registration for any cause it deems necessary."
            }}
        ]
        }}

        Repeat this process for every matrix provision. If the code is 1 or -1, ALWAYS cite the specific provision in the law where the provision from the matrix is found to be implemented or negated.
        Important: Remember to code 1 ONLY when the provision is EXPLICITLY IMPLEMENTED in the law, and -1 ONLY when it's EXPLICITLY NEGATED. If it's only inferred through lack of regulation or something else, it's a 0.

        The CSO Regulatory Regime Matrix is as follows: {matrix_typology}
    """
    start_time = time.time()

    response = client.responses.create(
        model = "gpt-4.1",
        instructions = system_instructions,
        input = prompt,
        temperature = 0
    )

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    usage = response.usage
    print("\nToken Usage:")
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Output tokens: {usage.output_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
    print(f"Execution time: {duration} seconds\n")

    return response.output_text