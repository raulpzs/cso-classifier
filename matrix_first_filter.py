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

def verify_provision(provision, law_text=None):
    prompt = f"""
    The first-round coder usually does a good job finding the section of the law that addresses the provision's topic, but tends to overuse the presence (1) or negation (-1) label instead of absence (0).
    In order for a provision to be labeled 1 or -1, it must be explicitly implemented or explicitly negated in the legal text, and all components of the ADICO structure from the matrix provision must be present (excluding the 'or else' clause, which is optional).

    In some instances, the provisions in the matrix do not regulate CSOs directly, but rather the regulating body, government, or other entities.
    However, the provision must be satisfied by the law text itself, even if it does not assign a specific duty to an agency.
    In these cases, the provision can still be correctly labeled as 1 or -1 if the legal text itself fulfills the obligation, even if it is the legislature performing the act rather than the agency, or an ADICO element is missing.

    Here are a few examples to guide your coding:

    a) INPUT:
    {{
    "Provision": "PERMISSIVE FORMATION 2. [Agency] [must] [explain precise legal definitions of CSOs that it regulates] [at law's commencement] [or else it is negligent in its duties]",
    "Code": 1,
    "Evidence": "Art. 19: 'Civil or commercial companies, public establishments of an industrial and commercial nature, cooperatives or mutual funds may create, with a view to the realization of a work of general interest, a not-for-profit legal person, called the corporate Foundation.'",
    "Explanation": "The law defines 'company Foundation' and its requirements."
    }}

    OUTPUT:
    {{
    "Provision": "PERMISSIVE FORMATION 2. [Agency] [must] [explain precise legal definitions of CSOs that it regulates] [at law's commencement] [or else it is negligent in its duties]",
    "FirstRoundCoderLabel": 1,
    "CorrectLabel": 1,
    "Evidence": "Art. 19: 'Civil or commercial companies, public establishments of an industrial and commercial nature, cooperatives or mutual funds may create, with a view to the realization of a work of general interest, a not-for-profit legal person, called the corporate Foundation.'",
    "Explanation": "The legal text itself defines the concept of a 'corporate foundation' at the outset, fulfilling the matrix provision's requirement that precise definitions be provided. Although this duty is not assigned to a specific agency, the obligation is satisfied by the statute itself. Therefore, the correct label is 1."
    }}

    b) INPUT:
    {{
    "Provision": "RESTRICTIVE FORMATION 1. [CSOs] [must not] [operate as informal, voluntary associations] and instead must register with the government [or else face penalty for non-compliance]",
    "Code": 1,
    "Evidence": "Art. 19-1: 'The company Foundation enjoys the legal capacity from the publication in the Official Gazette of the administrative authorization that him confers it this status.'",
    "Explanation": "Legal capacity is only granted upon publication of administrative authorization."
    }}
    
    OUTPUT
    And your output should be a JSON file that looks like this:
    {{
    "Provision": "RESTRICTIVE FORMATION 1. [CSOs] [must not] [operate as informal, voluntary associations] and instead must register with the government [or else face penalty for non-compliance]",
    "FirstRoundCoderLabel": 1,
    "CorrectLabel": 0,
    "Evidence": "Art. 19-1: 'The company Foundation enjoys the legal capacity from the publication in the Official Gazette of the administrative authorization that him confers it this status.'",
    "Explanation": "While Art. 19-1 requires administrative authorization and publication for legal capacity, the matrix provision is about prohibiting informal operation It does not explicitly forbid informal association, it only requires registration for legal recognition."
    }}

    c) INPUT:
    {{
    "Provision": " PERMISSIVE GOVERNANCE 2. [Government and agency] [must] [create or empower a dispute resolution forum] such as a court [before commencement of the law] [or else it is negligent in its duties]",
    "Code": 1,
    "Evidence": "Art. 9: 'The decision on the refusal of the state registration of a charitable organization may be appealed against with the court.'",
    "Explanation": "The law provides for judicial review of agency decisions."
    }}

    OUTPUT
    {{
    "Provision": "PERMISSIVE GOVERNANCE 2. [Government and agency] [must] [create or empower a dispute resolution forum] such as a court [before commencement of the law] [or else it is negligent in its duties]",
    "FirstRoundCoderLabel": 1,
    "CorrectLabel": 1,
    "Evidence": "Art. 9: 'The decision on the refusal of the state registration of a charitable organization may be appealed against with the court.'",
    "Explanation": "The law provides for judicial review of agency decisions, thus fulfilling the matrix provision's requirement for a dispute resolution forum."
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

def verify_provision_context(provision):
    prompt = f"""
    The first-round coder usually does a good job finding the section of the law that addresses the provision's topic, but tends to overuse the presence (1) or negation (-1) label instead of absence (0).
    In order for a provision to be labeled 1 or -1, it must be explicitly implemented or explicitly negated in the legal text, and all components of the ADICO structure from the matrix provision must be present (excluding the 'or else' clause, which is optional).

    In some instances, the provisions in the matrix do not regulate CSOs directly, but rather the regulating body, government, or other entities.
    However, the provision must be satisfied by the law text itself, even if it does not assign a specific duty to an agency.
    In these cases, the provision can still be correctly labeled as 1 or -1 if the legal text itself fulfills the obligation, even if it is the legislature performing the act rather than the agency, or an ADICO element is missing.

    Here are a few examples to guide your coding:

    a) INPUT:
    {{
    "Provision": "PERMISSIVE FORMATION 2. [Agency] [must] [explain precise legal definitions of CSOs that it regulates] [at law's commencement] [or else it is negligent in its duties]",
    "Code": 1,
    "Evidence": "Art. 19: 'Civil or commercial companies, public establishments of an industrial and commercial nature, cooperatives or mutual funds may create, with a view to the realization of a work of general interest, a not-for-profit legal person, called the corporate Foundation.'",
    }}

    OUTPUT:
    {{
    "Provision": "PERMISSIVE FORMATION 2. [Agency] [must] [explain precise legal definitions of CSOs that it regulates] [at law's commencement] [or else it is negligent in its duties]",
    "FirstRoundCoderLabel": 1,
    "CorrectLabel": 1,
    "Evidence": "Art. 19: 'Civil or commercial companies, public establishments of an industrial and commercial nature, cooperatives or mutual funds may create, with a view to the realization of a work of general interest, a not-for-profit legal person, called the corporate Foundation.'",
    "Explanation": "The legal text itself defines the concept of a 'corporate foundation' at the outset, fulfilling the matrix provision's requirement that precise definitions be provided. Although this duty is not assigned to a specific agency, the obligation is satisfied by the statute itself. Therefore, the correct label is 1."
    }}

    b) INPUT:
    {{
    "Provision": "RESTRICTIVE FORMATION 1. [CSOs] [must not] [operate as informal, voluntary associations] and instead must register with the government [or else face penalty for non-compliance]",
    "Code": 1,
    "Evidence": "Art. 19-1: 'The company Foundation enjoys the legal capacity from the publication in the Official Gazette of the administrative authorization that him confers it this status.'",
    }}
    
    OUTPUT
    And your output should be a JSON file that looks like this:
    {{
    "Provision": "RESTRICTIVE FORMATION 1. [CSOs] [must not] [operate as informal, voluntary associations] and instead must register with the government [or else face penalty for non-compliance]",
    "FirstRoundCoderLabel": 1,
    "CorrectLabel": 0,
    "Evidence": "Art. 19-1: 'The company Foundation enjoys the legal capacity from the publication in the Official Gazette of the administrative authorization that him confers it this status.'",
    "Explanation": "While Art. 19-1 requires administrative authorization and publication for legal capacity, the matrix provision is about prohibiting informal operation It does not explicitly forbid informal association, it only requires registration for legal recognition."
    }}

    c) INPUT:
    {{
    "Provision": " PERMISSIVE GOVERNANCE 2. [Government and agency] [must] [create or empower a dispute resolution forum] such as a court [before commencement of the law] [or else it is negligent in its duties]",
    "Code": 1,
    "Evidence": "Art. 9: 'The decision on the refusal of the state registration of a charitable organization may be appealed against with the court.'",
    }}

    OUTPUT
    {{
    "Provision": "PERMISSIVE GOVERNANCE 2. [Government and agency] [must] [create or empower a dispute resolution forum] such as a court [before commencement of the law] [or else it is negligent in its duties]",
    "FirstRoundCoderLabel": 1,
    "CorrectLabel": 1,
    "Evidence": "Art. 9: 'The decision on the refusal of the state registration of a charitable organization may be appealed against with the court.'",
    "Explanation": "The law provides for judicial review of agency decisions, thus fulfilling the matrix provision's requirement for a dispute resolution forum."
    }}

    === Matrix provision ===
    {provision['Provision']}

    === Coder's label ===
    {provision['Code']}

    === Citation ===
    {provision.get('Evidence')}

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
