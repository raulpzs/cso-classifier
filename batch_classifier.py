import os
import json
import time
import datetime
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# API key
load_dotenv()
client = OpenAI(api_key=os.getenv("api_key"))

filtered_csv_path = "outputs/filtered_provisions_KenyaPublicOrder.csv"
matrix_path = "data/cso-matrix.txt"

# Filtered provisions from provision_filter_llm.py
df = pd.read_csv(filtered_csv_path)
df = df[df["label"] == "provision"]

# Upload CSO Matrix to vector store
print(" Creating vector store...")
vector_store = client.vector_stores.create(name="CSO_Matrix_Store")
vector_store_id = vector_store.id

print(" Uploading matrix...")
with open(matrix_path, "rb") as f:
    client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store_id,
        files=[f]
    )

# Provision classifier (using file search only)
def classify_provision(provision_text, vector_store_id):
    instructions = f"""
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
    - Governance: Encompasses rules that shape how CSOs are managed internally, including leadership structures, decision-making, member rights, reporting obligations, and all procedures relating to administrative review, appeals, or contesting state actions against the organization'ss status. 
        This includes requirements to disclose governing body composition or changes in leadership to the state. Governance provisions regulate accountability and transparency within the organization, not its public-facing activities.
        Procedures for review or appeal of fines, suspension, or cancellation are classified under governance.
    - Operations: Regulates what CSOs are allowed to do in public life, including advocacy, service provision, protest, and communication with authorities. It includes rules about permits for projects, external coordination, and interaction with authorities about specific activities. Operations provisions affect the scope and conduct of daily actions in the civic sphere.
    - Resources: Pertains to CSO's ability to secure, manage, and use financial or material support, both domestic and international. This includes rules on fundraising, foreign funding, taxation, and ownership of assets.

    3. Classify the provision as either:
    - Restrictive: if it imposes barriers or burdens on CSO activity.
    - Permissive: if it enables, supports, or simplifies CSO activity.
    Consider a two-step reasoning process to classify:
        First, a provision is permissive if its reasonable and impartial enforcement improves trust, accountability, or resolves “voluntary failures”. 
        Classification advances to the second stage if there is no clear demand-side prediction. Here, a provision is restrictive if its reasonable and impartial enforcement limits organizational autonomy or stifles organizational emergence.
    If a provision has both elements, choose the effect that most strongly impacts the CSO's ability to act. If it grants a right with reasonable procedural steps, it is permissive; if it attaches a significant condition, penalty, or limitation, it is restrictive.

    Provide brief legal reasoning and justification of why it regulates in the chosen subgroup and type.

    Lock in the classification and subgroup assignment, do not change it later.

    4. Only then after lock in, you will use the CSO Regulatory Regime Matrix to match the closest listed concept only from the chosen subgroup and type.
    Match by both function and grammar (ADICO structure), considering legal substance and not just linguistic similarity.
    Do NOT look into provisions outside the subgroup and type you have previously classified the provision into.
    For example, if you classify a provision as "Governance" and "Restrictive", you will only look for matches in the Governance subgroup of the matrix, and only for Restrictive provisions.

    Within the subgroup, think about the primary, substantive area the provided provision regulates to generate a match. 
    For example, do not match to a provision about foreign funding unless the provision is mainly about receipt or use of foreign funds. For political activity or advocacy restrictions tied to tax/charitable status, match only to provisions related to political activity.

    If no exact match exists in the chosen subgroup, select the closest functional/grammatical equivalent from that subgroup, and flag this in "explanation".
    Remember concepts in the matrix are already classified into these subgroups, so the matched provision must be in accordance to the subgroup and type you have classified the provision into.
    Make sure there is not speculating beyond the matrix provided, the matched provision should alwasy exist in the CSO Regulatory Regime Matrix.

    Always return a JSON object with the following structure:

    {{
    "provision": "Provided provision text, exactly as it appears in the input.",
    "ADICO": "ADICO syntax breakdown of the provided provision.",
    "matched_matrix_provision": "[TYPE] [CATEGORY] Provision #[PROVISION NUMBER]: [Closest concept from subgroup, in the matrix, exactly as it appears in the matrix]",
    "subgroup": "Formation | Governance | Operations | Resources",
    "type": "Restrictive | Permissive",
    "explanation": "Brief legal reasoning and justification based on the matrix."
    }}

    """

    start_time = time.time()

    response = client.responses.create(
        model="gpt-4o",
        instructions=instructions,
        input=provision_text,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [vector_store_id]
        }],
        temperature=0.2
    )

    end_time = time.time()
    print(f" Classified in {round(end_time - start_time, 2)}s")
    return response.output_text

# Classify all provisions
results = []
for i, row in df.iterrows():
    provision = row["text"]
    print(f"\n Classifying provision {i} of {len(df)}")
    try:
        output = classify_provision(provision, vector_store_id)
        results.append({
            "provision": provision,
            "output": output
        })
    except Exception as e:
        print(f" Error: {e}")
        results.append({
            "provision": provision,
            "output": f"ERROR: {e}"
        })
    time.sleep(1.2)  # safe buffer

# Save results
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
output_path = f"outputs/classified_provisions_{timestamp}.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n Classification complete. Saved to {output_path}")