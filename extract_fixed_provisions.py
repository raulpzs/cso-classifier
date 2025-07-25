import pdfplumber
import csv, os, re, unicodedata
from typing import List
from ftfy import fix_text    
from unidecode import unidecode # 

# ----------------------------------------------------------------------
# 1)  Tiny, fast cleaner
# ----------------------------------------------------------------------
RE_CTRL  = re.compile(r'[\x00-\x1F\x7F]')
RE_ZW    = re.compile(r'[\u200B-\u200D\uFEFF]')

TRANSLATE_TABLE = str.maketrans({
    '“':'"', '”':'"',
    '‘':"'", '’':"'",
    '–':'-', '—':'-',
    '•':'-',
})

def clean_text(s: str, *, ascii_only: bool = False) -> str:
    """
    Fixes mojibake, normalises Unicode, strips control / zero-width chars,
    converts smart punctuation, collapses whitespace.  
    Set ascii_only=True if you need pure ASCII.
    """
    # common encoding mistakes → proper Unicode
    s = fix_text(s)

    # ligatures, fancy fractions, etc.
    s = unicodedata.normalize('NFKC', s)

    # punctuation tweaks & invisible cruft
    s = s.translate(TRANSLATE_TABLE)
    s = s.replace('\u00A0', ' ')
    s = RE_ZW.sub('', s)
    s = RE_CTRL.sub(' ', s)

    if ascii_only:
        s = unidecode(s)

    # collapse runs of whitespace
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

# ----------------------------------------------------------------------
# 2)  Your original function + in‑place cleaning
# ----------------------------------------------------------------------
def extract_provisions_by_spacing(
    pdf_path: str,
    output_csv: str,
    y_threshold: float = 15.0,
    min_words: int = 5,
    ascii_only: bool = False,            # <‑‑ NEW: toggle transliteration
):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    provisions: List[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            lines = page.extract_words(use_text_flow=True, keep_blank_chars=True)
            if not lines:
                continue

            # Sort lines by vertical position
            lines = sorted(lines, key=lambda l: l['top'])

            paragraph = ""
            last_y = None

            for line in lines:
                text_raw = line['text']
                if not text_raw.strip():
                    continue

                text = clean_text(text_raw, ascii_only=ascii_only)
                y    = line['top']

                if last_y is not None and abs(y - last_y) > y_threshold:
                    if len(paragraph.split()) >= min_words:
                        provisions.append(paragraph.strip())
                    paragraph = text
                else:
                    paragraph += " " + text

                last_y = y

            # commit the trailing paragraph on the page
            if len(paragraph.split()) >= min_words:
                provisions.append(paragraph.strip())

    # de‑duplicate while preserving order
    provisions = list(dict.fromkeys(provisions))

    # write the cleaned provisions
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "label"])
        for idx, prov in enumerate(provisions):
            writer.writerow([idx, prov, ""])  # label empty for manual tagging

    print(f"Extracted {len(provisions)} provisions → {output_csv}")

# ----------------------------------------------------------------------
# 3)  Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    extract_provisions_by_spacing(
        "inputs/FranceCP1990ENG.pdf",
        "outputs/provisions_from_spacing_fixed_france.csv",
        ascii_only=False,      # keep accents; set True for pure ASCII
    )
