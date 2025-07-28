import os
import json
from matrix_first_codings import code_law
from matrix_first_filter import extract_nonzero_provisions, verify_provision

# Configuration
LAW_PATH = "inputs/RussiaOnCharActRUS.txt"
OUTPUT_DIR = "outputs"
FIRST_ROUND_OUTPUT = os.path.join(OUTPUT_DIR, "RussiaOnCharActRUS_first_round.json")
VERIFIED_OUTPUT = os.path.join(OUTPUT_DIR, "RussiaOnCharActRUS_verified.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Load law text
    with open(LAW_PATH, "r", encoding="utf-8") as f:
        law_text = f.read()

    print(f"Processing: {LAW_PATH}")

    # First-round coding
    first_round_data = code_law(law_text)

    # Save first-round output
    with open(FIRST_ROUND_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(first_round_data, f, indent=2, ensure_ascii=False)

    print(f"Saved first-round coding to {FIRST_ROUND_OUTPUT}")

    # Filter and verify non-zero provisions
    nonzero_provisions = extract_nonzero_provisions(first_round_data)
    verified = []

    for provision in nonzero_provisions:
        print(f"Verifying: {provision['Provision'][:80]}...")
        verified_result = verify_provision(provision, law_text)
        verified.append(verified_result)

    # Save verified output
    with open(VERIFIED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(verified, f, indent=2, ensure_ascii=False)

    print(f"Saved verified provisions to {VERIFIED_OUTPUT}")

if __name__ == "__main__":
    main()