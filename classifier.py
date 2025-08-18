#!/usr/bin/env python3
import os
from openai import OpenAI
#from dotenv import load_dotenv
import time
import json

# Load .env variables (to keep the API key secure)
#load_dotenv()

if os.getenv("HF_SPACE") != "true":
    from dotenv import load_dotenv
    load_dotenv()

client = OpenAI(api_key=os.getenv("api_key"))

# Load the CSO Matrix typology from file
with open("data/cso-matrix.txt", "r", encoding="utf-8") as f:
    matrix_typology = f.read()

def classify_provision(provision_text):
    prompt = f"""
    Classify the following provision: {provision_text}
    """

    system_instructions = f"""
    You are a legal classification expert trained in civil society regulation. 
    Your task is to classify civil society regulations using the ADICO grammar and the CSO Regulatory Regime Matrix.

    1. You will break down the provided provision using the ADICO grammar as follows: 
    [ATTRIBUTES] identify to whom the statement applies, with the default assumption being all members of the group;
    [DEONTIC] identifies the expectation of behavior identified by the qualifiers 'may' (permitted), 'must' (obliged), and 'must not' (forbidden);
    [AIM] specifies the particular action or outcome prescribed, or those actions or outcomes that are forbidden;
    [CONDITIONS] explains when and where the institutional applies with the default assumption being all times and all places;
    [OR ELSE] assigns consequences for noncompliance. This component must have three qualifications: 
        (i) sanctioning provision is the result of an explicit collective-choice decision that is separate from any internal or social penalty
        (ii) be backed by at least one other institutional statement that if noncompliance occurs changes the DEONTIC assigned to some AIM for at least one actor
        (iii) affect the constraints and opportunities of actors responsible for monitoring the conformance of offenders.

    Example: The familiar example requiring American men to register for Selective Service can be written as an institutional statement thusly: 
    ATTRIBUTE [All male U.S. citizens between the ages of 18 and 25] 
    DEONTIC [must] 
    AIM [register for Selective Service within 30 days of their 18-birthday using one of the methods prescribed by the Selective Service System] 
    CONDITIONS [at all times and in all places unless they are exempted] 
    OR ELSE [or else face imprisonment, a ﬁne, or both].

    Not every provisison will have all components, especially [conditons] and [or else].
    If a component is not present, you can use "N/A" to indicate its absence, but you should always try to identify them before identifying as absent.
    
    2. Identify the provision to one of the four CSO Matrix subgroups depending on the substantive area it regulates: Formation, Governance, Operations, Resources.
    - Formation: Refers to the legal rules governing how CSOs come into legal existence, including registration, minimum founders, and procedures to gain formal recognition. It defines who is allowed to form a CSO and under what conditions.
    - Governance: Encompasses rules that shape how CSOs are managed internally, including leadership structures, decision-making, member rights, reporting obligations, and all procedures relating to administrative review, appeals, or contesting state actions against the organization's status. 
        This includes requirements to disclose governing body composition or changes in leadership to the state. Governance provisions regulate accountability and transparency within the organization, not its public-facing activities.
        Procedures for review or appeal of fines, suspension, or cancellation are classified under governance.
    - Operations: Regulates what CSOs are allowed to do in public life, including advocacy, service provision, protest, and communication with authorities. It includes rules about permits for projects, external coordination, and interaction with authorities about specific activities. 
        Operations provisions affect the scope and conduct of daily actions in the civic sphere.
    - Resources: Pertains to CSO's ability to secure, manage, and use financial or material support, both domestic and international. This includes rules on fundraising, foreign funding, taxation, and ownership of assets.

    A provision may regulate multiple subgroups, you may select more than one, but specify an order of priority if you do so.

    3. Classify the provision type as either:
    - Permissive if it enables, supports, or simplifies CSO activity.
    - Restrictive if it imposes barriers or burdens on CSO activity

    Consider a two-step reasoning process to classify:
        First, a provision is permissive if its reasonable and impartial enforcement improves trust, accountability, or resolves “voluntary failures”. 
        Classification advances to the second stage if there is no clear demand-side prediction. Here, a provision is restrictive if its reasonable and impartial enforcement limits organizational autonomy or stifles organizational emergence.

    If a provision has both elements, choose the effect that most strongly impacts the CSO's ability to act. 
    If it grants a right with reasonable procedural steps, it is permissive; if it attaches a significant condition, penalty, or limitation, it is restrictive.

    Provide brief legal reasoning and justification of why it regulates in the chosen subgroup and type.

    Lock in the subgroup and type, do not change it later.

    4. You will use the CSO Regulatory Regime Matrix to match the closest listed concept.
    Match by both function and grammar (ADICO structure), considering legal substance and not just linguistic similarity.
    Think about the primary, substantive area the provided provision regulates to generate a match. 

    For example, a provision might regulate the right to appeal charitable status and foreign funding, but its main topic is about the right to appeal, so in this case, it shouldn't be matched to a provision about foreign funding but rather about the right to petition or appeal.

    If no exact match exists, select the closest functional/grammatical equivalent, cosidering its main area of regulation, and flag this in "explanation".

    Remember concepts in the matrix are already classified into subgroups and type, so the matched provision must be in accordance to the subgroup and type you have classified the provision into.
    While there may be multiple subgroups the provided provision regulates, you should only match it to one provision in the matrix.
    Make sure there is not speculating beyond the matrix provided, the matched provision should alwasy exist in the CSO Regulatory Regime Matrix.

    Always return a JSON object with the following structure:

    {{
    "provision": "Provided provision text, exactly as it appears in the input.",
    "ADICO": "ADICO syntax breakdown of the provided provision.",
    "matched_matrix_provision": "[TYPE] [CATEGORY] Provision #[PROVISION NUMBER]: [Closest concept from subgroup, in the matrix, exactly as it appears in the matrix]",
    "subgroup": "Formation | Governance | Operations | Resources (may be more than one, but specify an order of priority)",
    "type": "Restrictive | Permissive",
    "explanation": "Brief legal reasoning and justification based on the matrix."
    }}

    The CSO Regulatory Regime Matrix is as follows: {matrix_typology}
    """

    start_time = time.time()

    response = client.responses.create(
        model = "gpt-4.1",
        instructions = system_instructions,
        input = prompt,
        temperature = 0.2
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

def classify_provision_with_file_search(provision_text, file_path):
    # Create vector store
    vector_store = client.vector_stores.create(name="CSO_Matrix_Vector_Store")
    vector_store_id = vector_store.id

    # Upload the CSO matrix
    with open(file_path, "rb") as f:
        client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=[f]
        )
    start_time = time.time()

    response = client.responses.create(
        model="gpt-4.1",
        instructions=f"""
        You are a legal classification expert trained in civil society regulation. 
        Your task is to classify civil society regulations using the ADICO grammar and the CSO Regulatory Regime Matrix.

        1. You will break down the provided provision using the ADICO grammar as follows: 
        [ATTRIBUTES] identify to whom the statement applies, with the default assumption being all members of the group;
        [DEONTIC] identifies the expectation of behavior identified by the qualifiers 'may' (permitted), 'must' (obliged), and 'must not' (forbidden);
        [AIM] specifies the particular action or outcome prescribed, or those actions or outcomes that are forbidden;
        [CONDITIONS] explains when and where the institutional applies with the default assumption being all times and all places;
        [OR ELSE] assigns consequences for noncompliance. This component must have three qualifications: 
            (i) sanctioning provision is the result of an explicit collective-choice decision that is separate from any internal or social penalty
            (ii) be backed by at least one other institutional statement that if noncompliance occurs changes the DEONTIC assigned to some AIM for at least one actor
            (iii) affect the constraints and opportunities of actors responsible for monitoring the conformance of offenders.

        Not every provisison will have all components, especially [conditons] and [or else].
        If a component is not present, you can use "N/A" to indicate its absence, but you should always try to identify them before identifying as absent.
        
        2. Identify the provision to one of the four CSO Matrix subgroups depending on the substantive area it regulates: Formation, Governance, Operations, Resources.
        - Formation: Refers to the legal rules governing how CSOs come into legal existence, including registration, minimum founders, and procedures to gain formal recognition. It defines who is allowed to form a CSO and under what conditions.
        - Governance: Encompasses rules that shape how CSOs are managed internally, including leadership structures, decision-making, member rights, reporting obligations, and all procedures relating to administrative review, appeals, or contesting state actions against the organization's status. 
            This includes requirements to disclose governing body composition or changes in leadership to the state. Governance provisions regulate accountability and transparency within the organization, not its public-facing activities.
            Procedures for review or appeal of fines, suspension, or cancellation are classified under governance.
        - Operations: Regulates what CSOs are allowed to do in public life, including advocacy, service provision, protest, and communication with authorities. It includes rules about permits for projects, external coordination, and interaction with authorities about specific activities. 
            Operations provisions affect the scope and conduct of daily actions in the civic sphere.
        - Resources: Pertains to CSO's ability to secure, manage, and use financial or material support, both domestic and international. This includes rules on fundraising, foreign funding, taxation, and ownership of assets.

        A provision may regulate multiple subgroups, you may select more than one, but specify an order of priority if you do so.

        3. Classify the provision type as either:
        - Permissive if it enables, supports, or simplifies CSO activity.
        - Restrictive if it imposes barriers or burdens on CSO activity

        Consider a two-step reasoning process to classify:
            First, a provision is permissive if its reasonable and impartial enforcement improves trust, accountability, or resolves “voluntary failures”. 
            Classification advances to the second stage if there is no clear demand-side prediction. Here, a provision is restrictive if its reasonable and impartial enforcement limits organizational autonomy or stifles organizational emergence.

        If a provision has both elements, choose the effect that most strongly impacts the CSO's ability to act. 
        If it grants a right with reasonable procedural steps, it is permissive; if it attaches a significant condition, penalty, or limitation, it is restrictive.

        Provide brief legal reasoning and justification of why it regulates in the chosen subgroup and type.

        Lock in the subgroup and type, do not change it later.

        4. You will use the CSO Regulatory Regime Matrix to match the closest listed concept.
        Match by both function and grammar (ADICO structure), considering legal substance and not just linguistic similarity.
        Think about the primary, substantive area the provided provision regulates to generate a match. 

        For example, a provision might regulate the right to appeal charitable status and foreign funding, but its main topic is about the right to appeal, so in this case, it shouldn't be matched to a provision about foreign funding but rather about the right to petition or appeal.

        If no exact match exists, select the closest functional/grammatical equivalent, cosidering its main area of regulation, and flag this in "explanation".

        Remember concepts in the matrix are already classified into subgroups and type, so the matched provision must be in accordance to the subgroup and type you have classified the provision into.
        While there may be multiple subgroups the provided provision regulates, you should only match it to one provision in the matrix.
        Make sure there is not speculating beyond the matrix provided, the matched provision should alwasy exist in the CSO Regulatory Regime Matrix.

        Always return a JSON object with the following structure:

        {{
        "provision": "Provided provision text, exactly as it appears in the input.",
        "ADICO": "ADICO syntax breakdown of the provided provision.",
        "matched_matrix_provision": "[TYPE] [CATEGORY] Provision #[PROVISION NUMBER]: [Closest concept from subgroup, in the matrix, exactly as it appears in the matrix]",
        "subgroup": "Formation | Governance | Operations | Resources (may be more than one, but specify an order of priority)",
        "type": "Restrictive | Permissive",
        "explanation": "Brief legal reasoning and justification based on the matrix."
        }}

        Always use the file search tool and look for terms, references, or legal concepts and get further clarification to look up the most relevant definitions or explanations.
        """,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [vector_store_id]
        }],
        input=( f"""
            Classify the following provision: {provision_text}
            """
        ),
        temperature=0,
        include=["file_search_call.results"]
    )

    end_time = time.time()
    print(f"Completed in {round(end_time - start_time, 2)} seconds.")

    usage = response.usage
    print("\nToken Usage:")
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Output tokens: {usage.output_tokens}")
    print(f"Total tokens: {usage.total_tokens}")

    file_search_snippets = [r.text for r in response.output[0].results]
    classification = json.loads(response.output[1].content[0].text)
    output = {
        "classification": classification,
        "file_search_snippets": file_search_snippets,
    }

    return output

def classify_provision_isolated(provision_text):
    prompt = f"""
    Classify the following provision: {provision_text}
    """

    system_instructions = f"""
    You are a legal classification expert trained in civil society regulation. 
    Your task is to classify civil society regulations using the Institutional Grammar.

    1. You will break down the provided provision using the ADICO Institutional Grammar as follows: 
    [ATTRIBUTES] identify to whom the statement applies, with the default assumption being all members of the group;
    [DEONTIC] identifies the expectation of behavior identified by the qualifiers 'may' (permitted), 'must' (obliged), and 'must not' (forbidden);
    [AIM] specifies the particular action or outcome prescribed, or those actions or outcomes that are forbidden;
    [CONDITIONS] explains when and where the institutional applies with the default assumption being all times and all places;
    [OR ELSE] assigns consequences for noncompliance. This component must have three qualifications: 
        (i) sanctioning provision is the result of an explicit collective-choice decision that is separate from any internal or social penalty
        (ii) be backed by at least one other institutional statement that if noncompliance occurs changes the DEONTIC assigned to some AIM for at least one actor
        (iii) affect the constraints and opportunities of actors responsible for monitoring the conformance of offenders.

    Not every provisison will have all components, especially [conditons] and [or else].
    If a component is not present, you can use "N/A" to indicate its absence, but you should always try to identify them before identifying as absent.

    2. Classify the provision to a subgroup and either restrictive or permissive according to the following:

    2.1. You will choose a subgroup, there may be multiple for a single provision, but specifiy and order of priority if you do so:
    - Governance provisions structure the amendment and enforcement of provisions contained in other subgroups. These provisions create and empower institutional actors such as government agencies, private self-regulators, and dispute resolution forums that affect CSOs externally. Laws that restrict constitutionally protected freedoms, such as the right to associate, are governance provisions because they relate to constitutional provisions that supersede legislative laws and policies. In effect, this category is superior to others because governance provisions control the creation, enforcement, and amendment of other regulatory provisions. 
    - Formation provisions are primarily concerned with the legal status and processes of voluntary associations that choose to incorporate as formalized CSOs. Whether informal associations must incorporate with the government is also a formation provision. These provisions stipulate the requirements for registration (e. g. membership, financial capital), how the registration process unfolds, and whether registrations expire. As a legal matter, the status of a CSO and the decision to become a formal organization may determine which provisions apply to it (e. g. lobbying, tax-deductible donations). As a political matter, these policies have a legitimizing effect on organizations and failure to secure/renew the proper status might lead to decreased assets from donors and suspicion from citizens.
    - Operations provisions regulate how CSOs deploy assets in pursuit of their organizational goals. At a high level, these provisions stipulate issue areas and establish what CSOs can or cannot do. Legal definitions and funding sources often define this operational space. For example, American 501(c)(3)s are limited in their ability to lobby, while nonprofit 501(c)(4)s have no such restriction. These provisions also outline whether and how CSOs must receive permission to conduct operations. The highest burden appears to be provisions that require CSOs to obtain a permit to perform specific projects, but less burdensome is the requirement that CSOs obtain a license to perform a general task. These provisions communicate what (if any) reporting CSOs are expected to make available and to whom.
    - Resources provisions govern the financial and non-financial assets of CSOs. Some studies consider only provisions that regulate if and how CSOs can receive foreign funding, or what refers to as “philanthropic protectionism". These include provisions that prohibit specific legal forms from engaging in fundraising altogether and others that permit CSOs to raise funds through business activities unrelated to their charitable missions. This subgroup includes provisions governing taxable activities, whether a CSO receives a tax-exemption, whether individuals who donate to a CSO receive a tax-deduction, and other similar matters. These provisions also discuss requirements for auditing and financial reporting, ownership of non-financial resources such as property and equipment, and expectations for working with local partners.

    2.2. You will classify provisions as either restrictive or permissive using a two-step process. First, a provision is permissive if the demand-side theory predicts the provision improves trust, accountability, or resolves “voluntary failures”. 
    Classification advances to the second stage if there is no clear demand-side prediction. Here, a provision is restrictive if the supply-side theory predicts it limits organizational autonomy or stifles organizational emergence.
    
    Restrictive provisions are those that deteriorate society's trust in CSOs thus decrease demand for such organizations, or they repress and intimidate organizations and their members and thus decrease the supply of CSOs. In the vernacular of transaction cost economics, restrictive provisions increase transaction costs and make it more costly to operate and create such organizations. In extreme instances, these policies legalize corrosive state action such as harassment and seizure of property, impose excessive burdens, restrict the freedom to associate, limit pluralism and stoke intolerance, and remove legal protections and due process.
    Permissive provisions are those that protect society and thus increase demand for CSOs, or they create and preserve CSOs and thereby increase the stock of CSOs overtime. Permissive provisions facilitate accountability and transparency that reduce transaction costs for users of nonprofit services. These policies encourage the development of a country's voluntary sector, allow CSOs to self-regulate and appeal regulators' decisions, provide legal rights and protections, permit access to funds and incentivize private donations, and facilitate practices thought to build trust and protect against misconduct by those that seek to abuse the rights and privileges of the legal form. Permissive provisions help to build trust for CSOs among the public and to prevent unscrupulous actors from abusing the legal form for private gain.

    Tie-breaker: If a provision appears to have both (a) demand-side benefits (e.g., greater trust, accountability, or resolution of a voluntary failure) and (b) supply-side costs (e.g., reduced autonomy, higher compliance costs, threat of state coercion), classify according to the dominant net effect:
    If the supply-side cost introduces or amplifies coercive sanctions (e.g., fines, licence revocation, dissolution, criminal liability) that materially raise CSOs' expected transaction costs, label the provision Restrictive.
    Otherwise, where facilitative benefits clearly outweigh residual compliance costs, label it Permissive.
    
    Provide brief legal reasoning and justification of why it regulates in the chosen subgroup and type.

    Always return a JSON object with the following structure:

    {{
    "provision": "Provided provision text, exactly as it appears in the input.",
    "ADICO": "ADICO syntax breakdown of the provided provision.",
    "subgroup": "Formation | Governance | Operations | Resources (may be more than one, but specify an order of priority)",
    "type": "Restrictive | Permissive",
    "explanation": "Brief legal reasoning and justification based on the matrix."
    }}
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