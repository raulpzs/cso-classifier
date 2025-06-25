import pdfplumber
import csv
import os
from typing import List

def extract_provisions_by_spacing(pdf_path: str, output_csv: str, y_threshold: float = 15.0, min_words: int = 5):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    provisions = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            lines = page.extract_words(use_text_flow=True, keep_blank_chars=True)
            if not lines:
                continue

            # Sort by y position
            lines = sorted(lines, key=lambda l: l['top'])
            paragraph = ""
            last_y = None

            for line in lines:
                y = line['top']
                text = line['text'].strip()
                if not text:
                    continue

                if last_y is not None and abs(y - last_y) > y_threshold:
                    # Commit current paragraph if valid
                    if len(paragraph.split()) >= min_words:
                        provisions.append(paragraph.strip())
                    paragraph = text
                else:
                    paragraph += " " + text

                last_y = y

            # Final paragraph on page
            if len(paragraph.split()) >= min_words:
                provisions.append(paragraph.strip())

    # Remove duplicates
    provisions = list(dict.fromkeys(provisions))

    # Write to CSV for labeling
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "label"])
        for idx, prov in enumerate(provisions):
            writer.writerow([idx, prov, ""])  # label empty for manual tagging

    print(f"Extracted {len(provisions)} provisions and saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    extract_provisions_by_spacing("inputs/StateofMexicoCSOLaw.pdf", "outputs/provisions_from_spacing_mexico.csv")

