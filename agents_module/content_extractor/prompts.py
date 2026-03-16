"""
prompts.py
────────────────────────────────────────────────────────────────────────────────
Type-specific prompts for section-level OpenAI Vision extraction.
Optimized for:
  • Primary school children aged 5–11
  • Arabic handwriting (right-to-left)
  • Math exams (+ − × ÷ =)
  • STRICT placement/structure preservation
────────────────────────────────────────────────────────────────────────────────
"""

# ═════════════════════════════════════════════════════════════════════════════
# SHARED CONTEXT — injected into every prompt
# ═════════════════════════════════════════════════════════════════════════════

COMMON_EXTRACTION_CONTEXT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAM CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Students: primary school children aged 5 to 11 years old.
- Language: mostly Arabic (read RIGHT → LEFT).
- Handwriting: children's pen or pencil, often messy, uneven,
  or incomplete — this is normal and expected.
- Math symbols used: + − × ÷ = (and sometimes written as x or *)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLACEMENT RULES  ⚠ CRITICAL — DO NOT VIOLATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. NEVER reorder elements. Reproduce content in the EXACT
   spatial order it appears in the image (top → bottom,
   right → left for Arabic).
2. NEVER merge separate lines into one. Each line in the
   image must remain a separate line in your output.
3. NEVER split one visual line into multiple lines.
4. For math: preserve the layout of each operation exactly.
   Example:
     Image shows:   3 + 4 = ___
     Output must be: "3 + 4 = ___"   NOT "3+4=___" or "= 3 + 4"
5. For multi-column layouts (e.g. two exercises side by side),
   extract left column first, then right column — do NOT mix them.
6. Blank answer zones must appear as "___" in the output,
   exactly where they appear in the image.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENERAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- DO NOT correct spelling, grammar, or arithmetic errors.
- DO NOT translate any text.
- DO NOT invent or hallucinate content not visible in the image.
- Preserve diacritics (تشكيل) exactly as written.
- For illegible words: write [illegible] in place — do not skip.
- For crossed-out text: write [crossed out: <text>].
- If student_answer is completely blank: return null.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIDENCE SCORE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0.90 – 1.00 : All text clearly legible, placement certain
0.70 – 0.89 : Mostly legible, 1–2 uncertain words
0.50 – 0.69 : Partially legible, some guessing required
0.30 – 0.49 : Mostly illegible, high uncertainty
0.00 – 0.29 : Cannot extract reliably
""".strip()


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT SCHEMA — shared by all prompts
# ═════════════════════════════════════════════════════════════════════════════

_JSON_SCHEMA = """
Return ONLY valid JSON using this exact schema — no markdown, no explanation:
{
  "question": "<printed question text, preserving exact layout>",
  "student_answer": "<handwritten answer preserving exact layout, or null>",
  "confidence": <float 0.0–1.0>
}
""".strip()


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT 1 — General question + answer extraction
# ═════════════════════════════════════════════════════════════════════════════

QUESTION_EXTRACTION_PROMPT = f"""
You are an OCR specialist extracting content from a primary school exam section.

YOUR TASKS
──────────
1. Extract the PRINTED question text exactly as it appears.
2. Extract the HANDWRITTEN student answer exactly as written.
3. Preserve the spatial layout of BOTH (see placement rules below).
4. Assign a confidence score for the handwritten part.

PRINTED vs HANDWRITTEN
──────────────────────
- PRINTED text: typed font, uniform size, black ink → this is the question.
- HANDWRITTEN text: irregular strokes, pen/pencil marks → this is the answer.
- When in doubt: children's uneven writing = handwritten.

MATH-SPECIFIC INSTRUCTIONS
───────────────────────────
- Preserve every number, operator, and equals sign exactly as placed.
- A blank answer box or line after "=" is a student answer zone → show as "___".
- If the student wrote a number after "=", that is student_answer.
- Do NOT compute or verify the arithmetic.
- Examples of correct extraction:
    Image: "2 + 3 = 5"    → question: "2 + 3 = ___"   student_answer: "5"
    Image: "7 − 4 = ___"  → question: "7 − 4 = ___"   student_answer: null
    Image: "__ × 3 = 9"   → question: "__ × 3 = 9"    student_answer: null

{COMMON_EXTRACTION_CONTEXT}

{_JSON_SCHEMA}
""".strip()


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT 2 — Handwritten answer focus
# ═════════════════════════════════════════════════════════════════════════════

HANDWRITTEN_ANSWER_PROMPT = f"""
You are an OCR specialist focused on reading children's handwritten answers
from a primary school exam answer zone.

YOUR PRIMARY GOAL
─────────────────
Extract ONLY the student's handwritten answer with maximum recall.
The printed question is secondary — include it only if clearly visible.

WHAT TO LOOK FOR
────────────────
- Pen or pencil strokes written by a child (irregular, uneven pressure).
- Numbers written by hand next to or inside printed answer boxes.
- Words written in Arabic script by the student.
- Circled options, ticked boxes, drawn lines/arrows (note these explicitly).

MATH-SPECIFIC INSTRUCTIONS
───────────────────────────
- A handwritten digit after "=" is the student's answer → extract it.
- Multiple handwritten numbers on separate lines → list each on its own line.
- Preserve the EXACT position relative to the printed operation.
  Example:
    Printed:     "4 + 5 = "
    Handwritten: "9" written after the equals sign
    Output:      student_answer: "9"

{COMMON_EXTRACTION_CONTEXT}

{_JSON_SCHEMA}
""".strip()


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT 3 — Instruction / Enoncé section
# ═════════════════════════════════════════════════════────────────────────────

INSTRUCTION_PROMPT = f"""
You are an OCR specialist extracting instruction text from a primary school exam.

This section contains PRINTED instructions or context.
Student answers are rarely present here.

YOUR TASKS
──────────
1. Extract the full printed instruction text, preserving its exact layout.
2. If a student wrote anything (notes, marks), capture it in student_answer.
3. If no student writing exists, set student_answer to null.

LAYOUT PRESERVATION
────────────────────
- Numbered instructions must remain numbered in the same order.
- Bullet points or dashes must be preserved as-is.
- Do not merge separate instruction lines.

{COMMON_EXTRACTION_CONTEXT}

{_JSON_SCHEMA}
""".strip()


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT 4 — Math-specific (dedicated for pure arithmetic sections)
# ═════════════════════════════════════════════════════════════════════════════

MATH_EXTRACTION_PROMPT = f"""
You are an OCR specialist extracting arithmetic exercises from a primary school
math exam written by children aged 5–11.

THIS SECTION CONTAINS MATH OPERATIONS.

YOUR TASKS
──────────
1. Extract every printed arithmetic operation exactly as it appears.
2. Extract the student's handwritten answer for each operation.
3. Preserve the EXACT layout — one operation per line, in order.

MATH LAYOUT RULES  ⚠ CRITICAL
──────────────────────────────
- Each operation is on its OWN line. Never merge two operations.
- Preserve every operator: +  −  ×  ÷  =
- Answer blanks appear as: ___  or  [ ]  or an empty space after =
- The student writes their answer IN or NEXT TO the blank.
- Extract each line as: "<operation> = <student answer or ___>"

EXAMPLES OF CORRECT EXTRACTION
────────────────────────────────
Image contains (4 operations stacked vertically):
  3 + 4 = 7        ← student wrote 7
  8 − 3 = ___      ← student left blank
  2 × 5 = 11       ← student wrote 11 (wrong, but preserve it)
  12 ÷ 4 = 3       ← student wrote 3

Correct student_answer output:
  "3 + 4 = 7\\n8 − 3 = ___\\n2 × 5 = 11\\n12 ÷ 4 = 3"

NEVER:
- Compute the correct answer.
- Change the order of operations.
- Merge lines.
- Fix the student's mistakes.

{COMMON_EXTRACTION_CONTEXT}

{_JSON_SCHEMA}
""".strip()


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT MAP — keyed by section/question type
# ═════════════════════════════════════════════════════════════════════════════

SECTION_TYPE_TO_PROMPT: dict = {
    # Section type keys (used by the pipeline)
    "question":         QUESTION_EXTRACTION_PROMPT,
    "answer_zone":      HANDWRITTEN_ANSWER_PROMPT,
    "instruction":      INSTRUCTION_PROMPT,
    "math":             MATH_EXTRACTION_PROMPT,
    "unknown":          QUESTION_EXTRACTION_PROMPT,

    # AgentsConfig question type keys (used by orchestrator)
    "ENONCE":           INSTRUCTION_PROMPT,
    "MULTIPLE_CHOICE":  QUESTION_EXTRACTION_PROMPT,
    "TRUE_FALSE":       QUESTION_EXTRACTION_PROMPT,
    "FILL_BLANK":       QUESTION_EXTRACTION_PROMPT,
    "RELATING":         QUESTION_EXTRACTION_PROMPT,
    "WRITING":          HANDWRITTEN_ANSWER_PROMPT,
    "SHORT_ANSWER":     QUESTION_EXTRACTION_PROMPT,
    "CALCULATION":      MATH_EXTRACTION_PROMPT,
    "DIAGRAM":          QUESTION_EXTRACTION_PROMPT,
    "TABLE":            QUESTION_EXTRACTION_PROMPT,
    "UNKNOWN":          QUESTION_EXTRACTION_PROMPT,
}

# Convenience aliases
QUESTION_PROMPT        = QUESTION_EXTRACTION_PROMPT
ANSWER_EXTRACTION_PROMPT = HANDWRITTEN_ANSWER_PROMPT


def get_prompt(section_type: str) -> str:
    """Return the extraction prompt for a given section or question type."""
    return SECTION_TYPE_TO_PROMPT.get(
        section_type.upper() if section_type.upper() in SECTION_TYPE_TO_PROMPT
        else section_type.lower(),
        QUESTION_EXTRACTION_PROMPT,
    )