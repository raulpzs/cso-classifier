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
        # INSTRUCTIONS
        You are a legal analyst trained to analyze civil society regulation. 
        You will code a legal regime by checking for the presence (1), absence (0), or negation (-1) of standard regulatory provisions.
        The provisions are from the CSO Regulatory Regime Matrix, which follow an ADICO syntax (Attributes, Deontic, Aim, Conditions, Or Else).

        TASK:
        1. Analyze the legal statute to understand its objective and what it regulates.
        2. For EACH of the 56 standardized provisions in the CSO Matrix (listed below):
        - Code as 1 if one or more sections of the law EXPLICITLY IMPLEMENT the rule.
        - Code as -1 if one or more sections of the law EXPLICITLY NEGATE or forbid the rule.
        - Code as 0 if the law is silent on the matter or does not clearly address the rule.

        If a provision is 1 or -1, you MUST include “evidence”, a verbatim quotation (60 words max) from the statute proving your decision.
        If no such quotation exists, you MUST label the provision 0, with a brief explanation when possible.
        When uncertain, pick 0.

        Important: The law may use a different terminology, but the matrix uses CSO (Civil Society Organization) as an umbrella term that includes terms such as charity, foundation, non-profit organizations and other kinds of voluntary associations.

        ## Step 1
        Before coding the law, provide:
            Name: Name of law (if available)
            Jurisdiction: Where the law applies
            Year: Year of enactment (if available)
            Language: The language the provided statute is in
            Objective: What the statute is intended to regulate, and if it's related to CSO activity

        ## Step 2
        Then, for each matrix provision, provide:
            Matrix provision with ID (exactly as it appears in matrix)
            Code: 1, 0, or -1
            Evidence: Verbatim quotation from the statute (60 words max) if applicable
            Brief explanation (1-2 sentences)

        You will receive a full statute text which is extensive, so you will need to read it carefully to find the relevant sections.
        Use these examples to guide your coding, the output shall be in a JSON file with the following structure:

        ## EXAMPLE
        ### INPUT: (will be the full text of the law, not shown here for brevity)
    
        ### OUTPUT: (only 4 provisions are shown here for brevity, but you will code all 56 provisions)
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
            "Evidence": "Art. 19-1: 'The company Foundation enjoys the legal capacity from the publication in the Official Gazette of the administrative authorization that him confers it this status.'",
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
            }},
            {{
            "Provision": "RESTRICTIVE RESOURCES 4. [CSOs] [must] [use certain depository institutions] such as government banks [after law's commencement] [or else face penalty for non-compliance] ",
            "Code": 0,
            "Evidence": "Art. 19-3: 'All securities must be placed in registered securities, in securities for which the nominative reference slip is provided for in the article 55 of Act No. 87-416 of 17 June, 1987 on savings or in securities accepted by the Banque de France as advance guarantees.'",
            "Explanation": "Article 19-3 merely limits the types of securities the foundation may hold. It does not direct the CSO to open accounts or deposit its funds in a particular bank or other designated depository institution."
            }}
        }}
        
        Remember that in order for a provision to be coded as 1 or -1, the rule must be clearly established in the law, and you MUST provide evidence from the law text; otherwise it's a 0.
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

def code_law_find_context(law_text):
    prompt = f"""
    Code the following law: {law_text}
    """

    system_instructions = f"""
        # INSTRUCTIONS
        You are a legal analyst trained to analyze civil society regulation. 
        You will code a legal regime by finding the relevant sections that relate to standard regulatory provisions.
        The provisions are from the CSO Regulatory Regime Matrix, which follow an ADICO syntax (Attributes, Deontic, Aim, Conditions, Or Else).
        Finally, you will check for the presence (1), absence (0), or negation (-1) of theseprovisions.

        TASK DESCRIPTION:
        1. Analyze the legal statute to understand its objective and what it regulates.
        2. For EACH of the 56 standardized provisions in the CSO Matrix (listed below):
        - Find all the relevant sections of the law that relate to the provision (100-250 words), then:
        - Code as 1 if implemented, -1 if negated, or 0 if not addressed.

        Important: The law may use a different terminology, but the matrix uses CSO (Civil Society Organization) as an umbrella term that includes terms such as charity, foundation, non-profit organizations and other kinds of voluntary associations.

        ## Step 1
        Before coding the law, provide:
            Name: Name of law (if available)
            Jurisdiction: Where the law applies
            Year: Year of enactment (if available)
            Language: The language the provided statute is in
            Objective: What the statute is intended to regulate, and if it's related to CSO activity

        ## Step 2
        ### 2.1 Evidence
        Find the relevant sections of the law that relate to the provision. 
        Include at least 100 words and maximum 250 words of the relevant sections of the law that relate to the provision.
        If the provision is not addressed in the statute at all, you may include an "N/A" in those and its code will always be 0.
        However, if there's a part in the law that relates to the provision, you MUST include that evidence.
        If the evidence needed to justify is less than 100 words, just include the surrounding context to make it at least 100 words.
         
        ### 2.2 Coding
        For each provision to which you found a relevant section, you will procede to classify as:
            - 1 if one or more sections of the law EXPLICITLY IMPLEMENT the rule.
            - -1 if one or more sections of the law EXPLICITLY NEGATE or forbid the rule.
            - 0 if the law does not clearly address the rule.
        Remember the provisions follow an ADICO syntax. Therefore you can use the elements to guide your coding.
        The attributes, deontic, aim and conditions must all be present in the law text for it to be coded as 1. When the deontic is negated, the code must be -1.

        For each provision you will provide:
            Matrix provision with ID (exactly as it appears in matrix)
            Code: 1, 0, or -1
            Evidence: Verbatim quotation from the statute (100-250 words) if applicable
            Brief explanation (1-2 sentences)

        You will receive a full statute text which is extensive, so you will need to read it carefully to find the relevant sections.
        Use these examples to guide your coding, the output shall be in a JSON file with the following structure:

        ## EXAMPLE
        ### INPUT: (will be the full text of the law, not shown here for brevity)
    
        ### OUTPUT: (only 4 provisions are shown here for brevity, but you will code all 56 provisions)
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
            "Evidence": "“Art. 19-8. The resources of the company Foundation include: 1 the payments of the founders, except for the initial endowment; 2 subsidies from the State, local authorities and their public institutions; 3 the remuneration for services rendered; 4 revenues from initial endowment and resources referred to in 1, 2 and 3 above. “Under penalty of withdrawal of the administrative authorization provided for in the article 19-1, the company Foundation cannot make call to the public generosity; it cannot receive donations nor bequests. “Art. 19-9. - Company foundations establish an annual balance sheet, an income statement and an annex. They appoint at least one auditor and one substitute, selected from the list mentioned in article 219 of law No. 66-537 of 24 July, 1966 on commercial companies, which perform their duties under the conditions laid down by the Act; the provisions of article 457of the law aforementioned are applicable to them. The penalties provided for in section 439 of the Act are applicable to the President and members of the boards of company foundations that will not, each year, established a balance sheet, an income statement and an annex. The provisions of articles 455 and 458 of the same law are also applicable to them.",
            "Explanation": "Art. 19-8 explicitly includes proceeds of services rendered under their resources, allowing them to participate in other business activities."
            }},
            {{
            "Provision": "RESTRICTIVE FORMATION 1. [CSOs] [must not] [operate as informal, voluntary associations] and instead must register with the government [or else face penalty for non-compliance]",
            "Code": 0,
            "Evidence": "Art. 19. - Civil or commercial companies, public establishments of an industrial and commercial nature, cooperatives or mutual funds may create, with a view to the realization of a work of general interest, a not for-profit legal person, called the corporate Foundation. At the time of the Constitution of the company Foundation, the founder or founders bring the initial allocation referred to in the article 19-6 and undertake to make the payments mentioned in the article19-7 of the present law. Art. 19-1. - The company Foundation enjoys the legal capacity from the publication in the Official Gazette of the administrative authorization that him confers it this status. Such authorization shall be deemed to have been granted by the end of four months from the date of the request. It is then made the subject of the publication as laid down in the paragraph above. The company Foundation shall make known to the administrative authority any changes made to its statutes; these changes are permitted in the same forms as the original statutes. Where the purpose of the amendment to the statutes is to increase the program of multi-annual action, the staffing must be completed in accordance with the article 19-6.",
            "Explanation": "While Art. 19-1 requires administrative authorization and publication for legal capacity, it does not explicitly forbid informal association, it only requires registration for legal recognition."
            }},
            {{
            "Provision": "PERMISSIVE OPERATIONS 4. [Agency] [must] [have reasonable cause and follow explicit rules when conducting inspections of CSOs] such as requesting specific documentation or investigating offenses [after law's commencement] [or else it is overstepping its authority] ",
            "Code": -1,
            "Evidence": "Art. 19-10. - The administrative authority shall ensure the regularity of the operation of the company Foundation; to this end, it can be given all documents and carry out any useful investigations. “The company Foundation shall, each year, address to the administrative authority a report on the status of its activities to which is attached the report of the auditor and annual accounts. “Art. 19-11.- When the Foundation is dissolved, either by the expiry of the period, or amicably by the withdrawal of all the founders, provided that they have fully paid the sums they have pledged to pay, a liquidator is appointed by the Board of Directors. If the Board has not been able to proceed with this nomination or if the dissolution results from the withdrawal of authorization, the liquidator is designated the judicial authority. The appointment of the liquidator is published in the Official Gazette. Art. 19-12. - In the event of...",
            "Explanation": "Article 19-10 states that the administrative authority may 'carry out any useful investigation', effectively removing the requirement for reasonable cause or specific rules."
            }},
            {{
            "Provision": "PERMISSIVE GOVERNANCE 1. [Agency] [must] [explain penalty for particular offenses, or explain a 'general penalty' for offenses where no penalty is expressly provided] [before commencement of the law] [or else it is negligent in its duties]",
            "Code": 1,
            "Evidence": "Art. 20.- Only recognized foundations of public utility may make use, in their title, their statutes, contracts, documents or publicity, of the designation of Foundation. However, may also be referred to as the irrevocable assignment, with a view to realization of general interest and non-profit, property, rights or resources to a recognized Foundation of public utility whose statutes have been approved as such, given that these assets, rights or resources are managed directly by the receiving Foundation, and without the creation of a separate legal entity for that purpose. Only company foundations that meet the conditions set out in sections 19-1 to 19-10 of this Act may use, in their title, their statutes, contracts, documents or publicity, the name of company Foundation. It may be accompanied by the name (s) of the founders. “Groupings constituted before the publication of this Act, which use in their denomination the terms of Foundation or of company Foundation, must comply to its provisions before 31 December, 1991. Presidents, directors or directors of groups who violate the provisions of this article shall be punished by a fine of 5000F to 15000F and, in the case of recidivism, by a fine of 10,000F  to 30,000F.",
            "Explanation": "Article 20 states that if a charitable organization receives repeated warnings in writing, it may be liquidated according to the procedure set out in the Civil Code, thus providing a general penalty for offenses where no specific penalty is provided."
            }},
            {{
            "Provision": "RESTRICTIVE RESOURCES 4. [CSOs] [must] [use certain depository institutions] such as government banks [after law's commencement] [or else face penalty for non-compliance] ",
            "Code": 0,
            "Evidence": "Art. 19-3. - The company Foundation may, subject to the provisions of article 19-8, make all acts of the civil life that is not prohibited by its statutes, but it cannot acquire or possess other buildings other than those necessary for the purpose it proposes. All securities must be placed in registered securities, in securities for which the nominative reference slip is provided for in the article 55 of Act No. 87-416 of 17 June, 1987 on savings or in securities accepted by the Banque de France as advance guarantees. Where the company Foundation holds shares of the founding companies or companies controlled by them, the Foundation may not exercise the voting rights attached to such shares. Art. 19-4.- The company Foundation is administered by a Board of Directors composed for not more than two thirds of the founders or their representatives and representatives of the staff, and for at least one third of the personalities qualified in its areas of intervention. The personalities are chosen by the founders or their representatives and appointed at the first constituent meeting of the Board of Directors.",
            "Explanation": "Article 19-3 merely limits the types of securities the foundation may hold. It does not direct the CSO to open accounts or deposit its funds in a particular bank or other designated depository institution."
            }}
        }}
        
        Remember that in order for a provision to be coded as 1 or -1, the rule must be clearly established in the law, and you MUST provide evidence from the law text; otherwise it's a 0.
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