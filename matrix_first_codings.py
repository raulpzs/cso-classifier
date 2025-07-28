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

schema = {
    "type": "object",
    "properties": {
        "Name":         {"type": "string"},
        "Jurisdiction": {"type": "string"},
        "Year":         {"type": ["integer", "null"]},
        "Language":     {"type": "string"},
        "Objective":    {"type": "string"},
        "Provisions": {
            "type": "array",
            "minItems": 56,
            "maxItems": 56,
            "items": {
                "type": "object",
                "properties": {
                    "Provision": {"type": "string"},
                    "Code": {"type": "integer", "enum": [-1, 0, 1]},
                    "Evidence": {"type": ["string", "null"]},
                    "Explanation": {"type": ["string", "null"]}
                },
                "required": ["Provision", "Code", "Evidence", "Explanation"],
                "additionalProperties": False
            }
        }
    },
    "required": [
        "Name", "Jurisdiction", "Year",
        "Language", "Objective", "Provisions"
    ],
    "additionalProperties": False
}


def code_law(law_text):
    prompt = f"""
    Code the following law: {law_text}
    """

    system_instructions = f"""
        You are a legal analyst trained to analyze civil society regulation. 
        You will code a legal regime by checking for the presence, absence, or negation of standard regulatory provisions.
        The provisions are from the CSO Regulatory Regime Matrix, which follow an ADICO syntax (Attributes, Deontic, Aim, Conditions, Or Else).

        TASK:
        1. Analyze the legal statute to understand its objective and what it regulates.
        2. For EACH of the 56 standardized provisions in the CSO Matrix (listed below):
        - Code as 1 if one or more sections of the law EXPLICITLY IMPLEMENT the rule.
        - Code as -1 if one or more sections of the law EXPLICITLY NEGATE or forbid the rule.
        - Code as 0 if the law is silent on the matter or does not clearly address the rule.

        If a provision is 1 or -1, you MUST include “evidence”, a verbatim quotation from the statute proving your decision.
        If no such quotation exists, you MUST label the provision 0, with a brief explanation when possible.
        When uncertain, pick 0.

        Important: The law may use a different terminology, but the matrix uses CSO (Civil Society Organization) as an umbrella term that includes terms such as charity, foundation, non-profit organizations and other kinds of voluntary associations.

        Before coding the law, provide:
            Name: Name of law (if available)
            Jurisdiction: Where the law applies
            Year: Year of enactment (if available)
            Language: The language the provided statute is in
            Objective: What the statute is intended to regulate, and if it's related to CSO activity

        Then, for each matrix provision, provide:
            Matrix provision with ID (exactly as it appears in matrix)
            Code: 1, 0, or -1
            Evidence: Verbatim quotation from the statute (60 words max) if applicable
            Brief explanation (1-2 sentences)

        Use this examples to guide your coding, the output shall be in a JSON file with the following structure:

        {{
        "Name": "Act No. 90-559",
        "Jurisdiction": "France",
        "Year": 1990,
        "Language": "French",
        "Objective": "This law establishes the legal framework for corporate foundations in France, allowing companies to create nonprofit entities to support public interest activities with defined governance and funding rules.",
        "Provisions": [
            {{
            "Provision": "PERMISSIVE RESOURCES 4. [CSOs] [may] [engage in unrelated business activities] such as revenue generation [after law's commencement] [or else choose not to pursue those activities]",
            "Code": 1,,
            "Evidence": "Art. 19-8: 'The resources of the company Foundation include [...] the remuneration for services rendered.'",
            "Explanation": "Art. 19-8 explicitly includes proceeds of services rendered under their resources, allowing them to participate in other business activities."
            }},
            {{
            "Provision": "RESTRICTIVE FORMATION 1. [CSOs] [must not] [operate as informal, voluntary associations] and instead must register with the government [or else face penalty for non-compliance]",
            "Code": 0,
            "Evidence": null,
            "Explanation": "While Art. 19-1 requires administrative authorization and publication for legal capacity, it does not explicitly forbid informal association, it only requires registration for legal recognition."
            }},
            {{
            "Provision": "PERMISSIVE OPERATIONS 4. [Agency] [must] [have reasonable cause and follow explicit rules when conducting inspections of CSOs] such as requesting specific documentation or investigating offenses [after law's commencement] [or else it is overstepping its authority] ",
            "Code": -1,
            "Evidence": "Art. 19-10: 'to this end, it can be given all documents and carry out any useful investigations.'",
            "Explanation": "Article 19-10 states that the administrative authority may 'carry out any useful investigation', effectively removing the requirement for reasonable cause or specific rules."
            }},
            {{
            "Provision": "PERMISSIVE GOVERNANCE 1. [Agency] [must] [explain penalty for particular offenses, or explain a 'general penalty' for offenses where no penalty is expressly provided] [before commencement of the law] [or else it is negligent in its duties]",
            "Code": 1,
            "Evidence": "Art. 20: 'In case of sending to it repeated warnings in written form, the charitable organization may be liquidated in conformity with the procedure, stipulated by the Civil Code.'",
            "Explanation": "Article 20 states that if a charitable organization receives repeated warnings in writing, it may be liquidated according to the procedure set out in the Civil Code, thus providing a general penalty for offenses where no specific penalty is provided."
            }}
        }}

        The CSO Regulatory Regime Matrix is as follows: {matrix_typology}

    """
    start_time = time.time()

    response = client.responses.create(
        model = "gpt-4.1",
        instructions = system_instructions,
        input = prompt,
        text={
                "format": {
                    "type": "json_schema",
                    "name": "law_Coding",
                    "strict": True,
                    "schema": schema
                }
            },
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

    return json.loads(response.output_text)