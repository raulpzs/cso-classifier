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
    Classify the following provision: {provision_text}
    Use the CSO Matrix to find the closest matching concept and classify the provision accordingly.
    If no exact match exists, choose the conceptually closest one, but flag it.

    Return a JSON object like this:

    {{
    "provision": "{provision_text}",
    "interpretation": "ADICO syntax breakdown of the provided provision.",
    "matched_matrix_provision": "Closest concept from matrix, exactly as it appears in the matrix.",
    "subgroup": "Formation | Governance | Operations | Resources",
    "type": "Restrictive | Permissive",
    "explanation": "Brief legal reasoning and justification based on the matrix."
    }}
    """

    system_instructions = f"""
    You are a legal classification expert trained in civil society regulation. 
    Your task is to classify civil society regulations using the ADICO grammar and the CSO Regulatory Regime Matrix.

    You will classify provisions based on the CSO Regulatory Regime Matrix.
    Each provision in the matrix is structured using the ADICO grammar as follows: 
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

    This is the matrix you will use to classify the provision: {matrix_typology}
    You fill find the closest matching concept in the matrix and classify the provideded provision accordingly.
    If no match exists, choose the conceptually closest one, but flag it.

    From the matched concept, you will:
    1. Classify the provision as either:
    - Restrictive: if it imposes barriers or burdens on CSO activity.
    - Permissive: if it enables, supports, or simplifies CSO activity.
    Consier a two-step reasoning process to classify:
        First, a provision is permissive if its reasonable and impartial enforcement improves trust, accountability, or resolves “voluntary failures”. 
        Classification advances to the second stage if there is no clear demand-side prediction. Here, a provision is restrictive if its reasonable and impartial enforcement limits organizational autonomy or stifles organizational emergence.
    2. Assign the provision to one of the four CSO Matrix subgroups: Formation, Governance, Operations, Resources.
    Remember concepts in the matrix are already classified into these subgroups, so the category must match the one in the matrix, as well as its permissive or restrictive nature.
    3. Provide a brief legal reasoning and justification.
    
    Once assigned, do not change the category.
    Do not speculate beyond the matrix provided, and always return a JSON object with the specified structure.
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

def classify_provision_with_file_search(provision_text, matrix_path):
    # Create vector store
    vector_store = client.vector_stores.create(name="CSO_Matrix_Vector_Store")
    vector_store_id = vector_store.id

    # Upload the CSO matrix
    print("Matriz uploaded")
    with open(matrix_path, "rb") as f:
        client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=[f]
        )

    # API
    print("\nClassifying...\n")
    start_time = time.time()

    response = client.responses.create(
        model="gpt-4.1",
        instructions=(f"""
            You are a legal classification expert trained in civil society regulation. 
            Your task is to classify civil society regulations using the ADICO grammar and the CSO Regulatory Regime Matrix.

            You will classify provisions based on the CSO Regulatory Regime Matrix.
            Each provision in the matrix is structured using the ADICO grammar as follows: 
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

            This is the matrix you will use to classify the provision: {matrix_typology}
            You fill find the closest matching concept in the matrix and classify the provideded provision accordingly.
            If no match exists, choose the conceptually closest one, but flag it.

            From the matched concept, you will:
            1. Classify the provision as either:
            - Restrictive: if it imposes barriers or burdens on CSO activity.
            - Permissive: if it enables, supports, or simplifies CSO activity.
            Consier a two-step reasoning process to classify:
                First, a provision is permissive if its reasonable and impartial enforcement improves trust, accountability, or resolves “voluntary failures”. 
                Classification advances to the second stage if there is no clear demand-side prediction. Here, a provision is restrictive if its reasonable and impartial enforcement limits organizational autonomy or stifles organizational emergence.
            2. Assign the provision to one of the four CSO Matrix subgroups: Formation, Governance, Operations, Resources.
            Remember concepts in the matrix are already classified into these subgroups, so the category must match the one in the matrix, as well as its permissive or restrictive nature.
            3. Provide a brief legal reasoning and justification.
            
            Once assigned, do not change the category.
            Do not speculate beyond the matrix provided, and always return a JSON object with the specified structure.
            """
        ),
        tools=[{
            "type": "file_search",
            "vector_store_ids": [vector_store_id]
        }],
        input=( f"""
            Classify the following provision: {provision_text}
            Use the CSO Matrix to find the closest matching concept and classify the provision accordingly.
            If no exact match exists, choose the conceptually closest one, but flag it.

            Return a JSON object like this:

            {{
            "provision": "{provision_text}",
            "interpretation": "ADICO syntax breakdown of the provided provision.",
            "matched_matrix_provision": "Closest concept from matrix, exactly as it appears in the matrix.",
            "subgroup": "Formation | Governance | Operations | Resources",
            "type": "Restrictive | Permissive",
            "explanation": "Brief legal reasoning and justification based on the matrix."
            }}
            """
        ),
        temperature=0.2
    )

    end_time = time.time()
    print(f"Completed in {round(end_time - start_time, 2)} seconds.")

    usage = response.usage
    print("\nToken Usage:")
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Output tokens: {usage.output_tokens}")
    print(f"Total tokens: {usage.total_tokens}")

    return response.output_text