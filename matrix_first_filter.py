import os
import json
from openai import OpenAI

if os.getenv("HF_SPACE") != "true":
    from dotenv import load_dotenv
    load_dotenv()

client = OpenAI(api_key=os.getenv("api_key"))

def extract_nonzero_provisions(record: dict) -> list[dict]:
    """Return only provisions whose Code is ±1."""
    return [p for p in record.get("Provisions", []) if p.get("Code", 0) != 0]

schema = {
    "type": "object",
    "properties": {
        "Provision": {"type": "string"},
        "FirstRoundCoderLabel": {"type": "integer", "enum": [-1, 0, 1]},
        "CorrectLabel": {"type": "integer", "enum": [-1, 0, 1]},
        "Evidence": {"type": ["string", "null"]},
        "Explanation": {"type": ["string", "null"]}
    },
    "required": ["Provision", "FirstRoundCoderLabel", "CorrectLabel", "Evidence", "Explanation"],
    "additionalProperties": False
}

def verify_provision(provision, law_text):
    prompt = f"""
    You will see the law from where the first-round coder assinged its label, but pay special attention to where it cited from to assign said label.
    In order for a provision to be 1 or -1, it has to be explicitly implemented or negated, and all components of the ADICO from the matrix provision have to be present (except the 'or else' clause).

    You will see something like this:
    {{
    "Provision": "2. RESTRICTIVE FORMATION 1. [CSOs] [must not] [operate as informal, voluntary associations] and instead must register with the government [or else face penalty for non-compliance]",
    "Code": 1,
    "Evidence": "Art. 19-1: 'The company Foundation enjoys the legal capacity from the publication in the Official Gazette of the administrative authorization that him confers it this status.'",
    "Explanation": "Legal capacity is only granted upon publication of administrative authorization."
    }}

    And your output should be a JSON file that looks like this:
    {{
    "Provision": "2. RESTRICTIVE FORMATION 1. [CSOs] [must not] [operate as informal, voluntary associations] and instead must register with the government [or else face penalty for non-compliance]",
    "FirstRoundCoderLabel": 1,
    "CorrectLabel": 0,
    "Evidence": "Art. 19-1: 'The company Foundation enjoys the legal capacity from the publication in the Official Gazette of the administrative authorization that him confers it this status.'",
    "Explanation": "The matrix provision is about prohibiting and establishing a penalty for informal operation. While Art. 19-1 requires administrative authorization and publication for legal capacity, it does not explicitly forbid informal association, it only requires registration for legal recognition."
    }}

    === Matrix provision ===
    {provision['Provision']}

    === Coder's label ===
    {provision['Code']}

    === Citation ===
    {provision.get('Evidence')}

    === Explanation ===
    {provision.get('Explanation')}

    === Law text ===
    {law_text}
    
    """
    system_instructions = f"""
    You are a legal analyst trained to analyze civil society regulation.
    You will code a legal regime by checking for the presence (1), absence (0), or negation (-1) of standard regulatory provisions.
    The provisions are from the CSO Regulatory Regime Matrix, which follow an ADICO syntax (Attributes, Deontic, Aim, Conditions, Or Else).
    You are checking whether the first-round coder assigned the correct label to the provision below.
    """

    response = client.responses.create(
        model="o3",
        instructions = system_instructions,
        input = prompt,
        text={
                "format": {
                    "type": "json_schema",
                    "name": "law_verifier",
                    "strict": True,
                    "schema": schema
                }
            },
        )
    return json.loads(response.output_text)
