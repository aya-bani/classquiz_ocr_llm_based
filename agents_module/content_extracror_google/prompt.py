"""
prompt.py — LLM post-processing prompts for Arabic handwriting OCR.

These prompts are used AFTER Google Vision has already run OCR.
The LLM's job here is purely post-processing:
  • Decide which Vision-extracted fragments are student handwriting
  • Remove printed template text that Vision mixed in
  • Return the clean student answer without guessing or filling gaps

The LLM does NOT see the image — it sees only the raw Vision text
output and the spatial metadata we pass to it.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — sent once per session / request
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a post-processing assistant for an Arabic handwriting OCR system
used on primary school exams from Tunisia.

Google Cloud Vision has already extracted raw text from exam images.
Your job is to clean that raw text by:
  1. Keeping only the student's handwritten Arabic answer
  2. Removing printed exam template text
  3. Returning an empty string if no valid student answer is found

STRICT RULES — follow all of them without exception:

WHAT TO KEEP
────────────
• Arabic words or digits that are part of the student's answer
• Partially spelled words (children make spelling errors — keep them)
• Numbers written by the student (even if arithmetically wrong)
• Single letters or short words if they appear in the answer zone

WHAT TO REMOVE
──────────────
• Printed question text (usually starts with: اختر، اربط، أكمل، احسب،
  ارسم، اكتب، علل، فسر، السؤال، تعليمة, or similar instruction verbs)
• Repeated placeholder characters: "......" "------" "______" "ـــــ"
• Section numbers, page numbers, decorative borders
• The words of the question itself — only the answer matters
• Any text you are not confident belongs to the student

WHAT TO NEVER DO
────────────────
• NEVER complete a partial word or answer
• NEVER guess what the student intended to write
• NEVER correct spelling or grammar
• NEVER translate anything
• NEVER add text that was not in the Vision output
• NEVER rearrange the student's words

OUTPUT FORMAT
─────────────
Return ONLY the cleaned student answer as a plain string.
• No JSON, no markdown, no explanation.
• If nothing valid remains after filtering → return exactly: ""
• Preserve right-to-left Arabic word order exactly as received.
• Preserve original Arabic spelling, including errors.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# USER PROMPT TEMPLATE — filled per extraction call
# ─────────────────────────────────────────────────────────────────────────────

USER_PROMPT_TEMPLATE = """
Below is the raw text extracted by Google Cloud Vision from one section
of an Arabic primary school exam. The text may contain a mix of:
  • The printed exam question / instructions
  • The student's handwritten answer
  • Placeholder characters from the exam template

Section type : {section_type}
Question text: {question_text}

Raw Vision output:
──────────────────
{raw_ocr_text}
──────────────────

Your task:
Remove all printed question text and template placeholders.
Return ONLY the student's handwritten answer.
If no handwritten answer is present, return exactly: ""
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# STRICT VARIANT — used when confidence is borderline
# ─────────────────────────────────────────────────────────────────────────────

USER_PROMPT_STRICT = """
Below is low-confidence OCR output from an Arabic primary school exam.
The handwriting may be faint, messy, or partially illegible.

Section type : {section_type}
Question text: {question_text}

Raw Vision output (low confidence — treat with extra caution):
──────────────────────────────────────────────────────────────
{raw_ocr_text}
──────────────────────────────────────────────────────────────

STRICT MODE RULES:
• Only extract text you are certain is a handwritten student answer.
• If the text is a mix of printed and handwritten and you cannot
  clearly separate them → return "".
• If ANY part of the answer is unclear → return "" for that part.
  Do not guess. Do not interpolate.
• A partial answer (a single word or digit) is acceptable.
• An empty string is always acceptable and preferred over a wrong answer.

Return ONLY the student's handwritten answer, or "".
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# DOTTED-LINE VARIANT — used when answer zone is a blank/dotted line
# ─────────────────────────────────────────────────────────────────────────────
# This is the most common layout: the exam prints "السؤال: ............"
# and the student writes on top of the dots.
# Vision often returns the dots AND the handwriting mixed together.

USER_PROMPT_DOTTED_LINE = """
This exam section uses a DOTTED-LINE answer format.
The exam template prints a line of dots ("............") and the student
writes their answer directly ON TOP of those dots.

Google Vision extracted the following text from that area (dots and
handwriting are mixed together in the raw output):

Section type : {section_type}
Question text: {question_text}

Raw Vision output (dots + handwriting mixed):
─────────────────────────────────────────────
{raw_ocr_text}
─────────────────────────────────────────────

Your task:
1. Remove all the printed dots ("....." or similar repeated characters).
2. Remove the printed question text if it appears.
3. Extract ONLY the irregular Arabic text written by the student ON the dots.
4. The student's handwriting appears as Arabic words or digits that do NOT
   look like part of the uniform dot pattern.

If only dots remain after filtering → return "".
If handwriting is present → return it exactly as extracted.
Do not guess, complete, or correct anything.

Return ONLY the student's handwritten answer, or "".
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: pick the right prompt variant
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(
    raw_ocr_text: str,
    section_type: str = "unknown",
    question_text: str = "",
    avg_confidence: float = 1.0,
    has_dotted_line: bool = False,
) -> str:
    """Return the best user prompt for the given extraction context.

    Selection logic:
    1. dotted-line format → DOTTED_LINE variant (addresses the main bug)
    2. confidence < 0.55  → STRICT variant (extra conservative)
    3. otherwise          → standard variant
    """
    ctx = {
        "raw_ocr_text":  raw_ocr_text  or "(empty)",
        "section_type":  section_type  or "unknown",
        "question_text": question_text or "(not provided)",
    }

    if has_dotted_line:
        return USER_PROMPT_DOTTED_LINE.format(**ctx)
    if avg_confidence < 0.55:
        return USER_PROMPT_STRICT.format(**ctx)
    return USER_PROMPT_TEMPLATE.format(**ctx)