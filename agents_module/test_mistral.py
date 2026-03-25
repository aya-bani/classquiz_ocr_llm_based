import os
import json
import base64
import re
from pathlib import Path
from datetime import datetime
from mistralai.client import Mistral

# ============================================
# CONFIGURATION
# ============================================

api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    print("❌ MISTRAL_API_KEY not found")
    exit(1)

client = Mistral(api_key=api_key)

INPUT_FOLDER = "Exams/sections/math"
OUTPUT_FOLDER = "Exams/extraction_results"

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# ============================================
# SECTION 1 — OCR PRE-PROCESSING
# Cleans raw Mistral OCR markdown before it
# is sent to the LLM for answer extraction.
# ============================================

# Arabic Unicode character range
_ARABIC_RANGE = r'\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF'

# Common OCR character-level fixes
_OCR_CHAR_FIXES: list[tuple[str, str]] = [
    ('O', '0'),      # capital O → zero (digit-context only — see _fix_digit_lookalikes)
    ('l', '1'),      # lowercase L → one (digit-context only)
    ('٪', '%'),      # Arabic percent → ASCII
    ('−', '-'),      # Unicode minus → ASCII minus
    ('×', 'x'),      # multiplication sign → x
    ('÷', '/'),      # division sign → /
    ('＝', '='),     # full-width equals → ASCII equals
    ('\u200b', ''),  # zero-width space
    ('\u200c', ''),  # zero-width non-joiner
    ('\u200d', ''),  # zero-width joiner
    ('\ufeff', ''),  # BOM
    ('\u202a', ''),  # LEFT-TO-RIGHT EMBEDDING
    ('\u202b', ''),  # RIGHT-TO-LEFT EMBEDDING
    ('\u202c', ''),  # POP DIRECTIONAL FORMATTING
    ('\u202d', ''),  # LEFT-TO-RIGHT OVERRIDE
    ('\u202e', ''),  # RIGHT-TO-LEFT OVERRIDE
    ('\u2066', ''),  # LEFT-TO-RIGHT ISOLATE
    ('\u2067', ''),  # RIGHT-TO-LEFT ISOLATE
    ('\u2068', ''),  # FIRST STRONG ISOLATE
    ('\u2069', ''),  # POP DIRECTIONAL ISOLATE
]

# Markdown artefacts Mistral OCR sometimes injects
_MARKDOWN_NOISE = re.compile(
    r'(\*{1,3}|_{1,3}|#{1,6}\s?|`{1,3}|\[.*?\]\(.*?\)|\!\[.*?\]\(.*?\))',
    re.UNICODE,
)
_TABLE_SEPARATOR = re.compile(r'^\|?[\s\-|]+\|?\s*$', re.MULTILINE)
_RULE_LINE       = re.compile(r'^[\-=_\s]{3,}$', re.MULTILINE)
_DIGIT_BORDER    = re.compile(r'(?<=\d)[Ol](?=\d)|(?<=\d)[Ol]$|^[Ol](?=\d)', re.MULTILINE)


def _fix_digit_lookalikes(text: str) -> str:
    """Replace O→0 and l→1 ONLY when flanked by digits."""
    def _replace(m: re.Match) -> str:
        return '0' if m.group(0) == 'O' else '1'
    return _DIGIT_BORDER.sub(_replace, text)


def _normalize_arabic(text: str) -> str:
    """Normalize common Arabic glyph variants that OCR confuses."""
    # alef maqsura (ى) is frequently misread as ya (ي)
    text = text.replace('\u0649', '\u064a')
    return text


def preprocess_ocr_text(raw_text: str) -> str:
    """
    Clean raw Mistral OCR markdown before sending to the LLM.

    Steps:
      1. Strip bidi control characters
      2. Strip Markdown formatting artefacts
      3. Remove table-separator rows
      4. Remove decorative rule lines
      5. Apply char-level operator/digit fixes
      6. Fix digit lookalikes in numeric context only
      7. Normalize Arabic glyph variants
      8. Collapse excessive blank lines
    """
    if not raw_text:
        return raw_text

    text = raw_text

    # 1. Bidi control characters first
    _BIDI = {'\u200b', '\u200c', '\u200d', '\ufeff',
             '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
             '\u2066', '\u2067', '\u2068', '\u2069'}
    for old, new in _OCR_CHAR_FIXES:
        if old in _BIDI:
            text = text.replace(old, new)

    # 2. Strip Markdown formatting
    text = _MARKDOWN_NOISE.sub('', text)

    # 3. Remove table separators
    text = _TABLE_SEPARATOR.sub('', text)

    # 4. Remove decorative rule lines
    text = _RULE_LINE.sub('', text)

    # 5. Remaining char-level fixes (operators, percent, etc.)
    for old, new in _OCR_CHAR_FIXES:
        if old not in ('O', 'l'):
            text = text.replace(old, new)

    # 6. Digit lookalikes in numeric context only
    text = _fix_digit_lookalikes(text)

    # 7. Normalize Arabic glyph variants
    text = _normalize_arabic(text)

    # 8. Collapse 3+ blank lines → one blank line
    text = re.sub(r'\n{3,}', '\n\n', text)

    lines = [ln.rstrip() for ln in text.split('\n')]
    return '\n'.join(lines).strip()


# ============================================
# SECTION 2 — ENHANCED LLM PROMPT BUILDER
# Produces a richer system + user prompt pair.
# ============================================

_ENHANCED_SYSTEM_PROMPT = """\
You are an advanced OCR reasoning engine specialized in handwritten Arabic exam analysis.
You do NOT behave like a simple text extractor.
You simulate how a human expert reads, interprets, and corrects handwritten student answers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Do NOT hallucinate answers
• Do NOT blindly trust OCR text
• Always validate before extracting
• If a section is blank → return empty string, never invent

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR REASONING PROCESS — FOLLOW EVERY STEP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — RAW OCR INTERPRETATION
Treat the OCR text as noisy, possibly incorrect data. Identify:
  • Numbers (may be Arabic-Indic: ٠١٢٣٤٥٦٧٨٩ → convert to 0-9)
  • Arabic text (RTL, preserve word order)
  • Mathematical operators: = + - x /
  • Noise artefacts to ignore: ---, | |, markdown symbols, stray punctuation

STEP 2 — DIRECTION & STRUCTURE ANALYSIS
  • Arabic text   → RIGHT-TO-LEFT, preserve as written
  • Numbers       → LEFT-TO-RIGHT within their group
  • Math expressions may be written visually RTL by the student
    Example: student writes "2925 = 6705 - 9630"
    Do NOT assume this is already correct — proceed to Step 3.

STEP 3 — MATHEMATICAL VALIDATION (CRITICAL)
For every detected equation, do the following:
  1. Evaluate it as written
  2. If the result is mathematically INCORRECT:
       a. Try reversing the operands
       b. Try alternative operator interpretations
       c. Select the version that is mathematically correct
  3. Return the corrected, intended answer

  Example reasoning:
    OCR gives:  2925 = 6705 - 9630
    Check:      6705 - 9630 = -2925  ❌ incorrect
    Reversed:   9630 - 6705 = 2925   ✅ correct
    Conclusion: student intended 2925 = 9630 - 6705

  Another example:
    OCR gives:  15 = 3 x 5
    Check:      3 x 5 = 15  ✅ correct — keep as-is

  If no valid interpretation exists → mark validated: false, notes: "uncertain"

STEP 4 — HANDWRITING BEHAVIOR HANDLING
  • Ignore crossed-out values — prefer the final visible answer
  • Handle answers written on dotted lines inline
  • Detect and apply student self-corrections

STEP 5 — NOISE CORRECTION
  • Fix obvious OCR digit errors (e.g. 0 vs O, 1 vs l, 6 vs b)
  • Preserve mathematical meaning, not raw characters
  • If genuinely uncertain → set interpreted_answer: "uncertain"

STEP 6 — DIGIT & OPERATOR NORMALISATION
  • Use ONLY Western digits: 0 1 2 3 4 5 6 7 8 9
  • Use ONLY these operators: + - x / =
  • NO decimal points inside whole numbers (2089 not 20.89)
  • NO pipe symbols | anywhere

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a single valid JSON object. No markdown fences. No prose.

Schema:
{
  "student_answer": "<final validated answer as string, multi-line uses \\n>",
  "confidence": <float 0.0-1.0>,
  "validated": <true if math checks out, false if uncertain>,
  "notes": "<explain any correction made, or empty string if none>"
}

FORBIDDEN in output:
• NO pipe symbols |
• NO vertical bars or table separators
• NO invented content
• NO markdown inside JSON values
• NO nested JSON inside student_answer
"""

_ENHANCED_USER_TEMPLATE = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLEANED OCR TEXT FROM EXAM IMAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ocr_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK: Extract the student's handwritten answer.
• Preserve RTL order exactly as written.
• Convert Arabic-Indic digits to Western (0-9).
• Return ONLY valid JSON — no extra text.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Student Answer JSON:
"""


def build_enhanced_prompt(section_type: str, ocr_text: str) -> tuple[str, str]:
    """
    Build an improved (system_prompt, user_prompt) pair for LLM extraction.

    Args:
        section_type: e.g. "answer_zone" — passed through from caller.
        ocr_text:     Already preprocessed OCR text.

    Returns:
        (system_prompt, user_prompt) tuple ready for client.chat.complete().
    """
    user_prompt = _ENHANCED_USER_TEMPLATE.format(ocr_text=ocr_text)
    return _ENHANCED_SYSTEM_PROMPT, user_prompt


# ============================================
# SECTION 2b — MATH VALIDATION ENGINE
# Verifies and corrects extracted equations
# before they reach the output layer.
# Mirrors Step 3 of the reasoning prompt as
# a Python-side safety net.
# ============================================

# Matches:  2925 = 9630 - 6705  or  15 = 3 x 5
_EQ_PATTERN = re.compile(r'(\d+)\s*=\s*(\d+)\s*([+\-x/])\s*(\d+)')


def _eval_expr(a: int, op: str, b: int):
    """Evaluate a op b. Returns None on division by zero or unknown op."""
    if op == '+': return a + b
    if op == '-': return a - b
    if op == 'x': return a * b
    if op == '/': return a // b if b != 0 else None
    return None


def _try_fix_equation(result: int, a: int, op: str, b: int):
    """
    Given result = a op b that failed, try all valid reorderings.
    Returns a corrected equation string, or None if no fix found.
    """
    # Try reversed operands: result = b op a
    if _eval_expr(b, op, a) == result:
        # For commutative ops (+, x), only reorder if it actually differs
        reordered = f"{result} = {b} {op} {a}"
        if reordered != f"{result} = {a} {op} {b}":
            return reordered
        return f"{result} = {a} {op} {b}"  # already canonical
    # Try: a = result op b
    if _eval_expr(result, op, b) == a:
        return f"{a} = {result} {op} {b}"
    # Try: b = a op result
    if _eval_expr(a, op, result) == b:
        return f"{b} = {a} {op} {result}"
    return None


def validate_math_in_answer(answer: str) -> tuple:
    """
    [NEW] Scan the answer for equations and mathematically validate each one.

    For every equation of the form  R = A op B:
      1. Check if A op B == R
      2. If not, try all valid reorderings (RTL correction)
      3. Replace the incorrect form with the corrected one in the answer

    Args:
        answer: Cleaned student answer string.

    Returns:
        (corrected_answer, all_valid: bool, notes: str)
    """
    if not answer:
        return answer, True, ""

    corrections = []
    all_valid   = True
    result_text = answer

    for m in _EQ_PATTERN.finditer(answer):
        r, a, op, b = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
        computed = _eval_expr(a, op, b)

        if computed == r:
            continue  # Already correct

        fix = _try_fix_equation(r, a, op, b)
        if fix:
            result_text = result_text.replace(m.group(0), fix, 1)
            corrections.append(f"'{m.group(0)}' corrected to '{fix}'")
        else:
            all_valid = False
            corrections.append(f"'{m.group(0)}' could not be validated")

    notes = "; ".join(corrections) if corrections else ""
    return result_text, all_valid, notes


# ============================================
# SECTION 3 — ROBUST JSON VALIDATOR / REPAIRER
# Recovers valid data from malformed LLM JSON.
# ============================================

_TRAILING_COMMA = re.compile(r',\s*([}\]])')


def _last_resort_extract(text: str) -> str:
    """Pull a bare student answer via regex when JSON is completely broken."""
    m = re.search(r'student_answer["\s:]+([^\n"]+)', text)
    if m:
        return m.group(1).strip().strip('"').strip("'").replace('|', '')
    return ""


def validate_and_repair_json(response_text: str) -> dict:
    """
    Robustly parse and repair JSON returned by the LLM.

    Handles:
      • Markdown code fences
      • Trailing commas before } or ]
      • Nested JSON string inside student_answer
      • Missing student_answer key
      • Pipe symbol contamination in values
      • Confidence coercion to float in [0, 1]
    """
    text = response_text.strip()

    # Strip markdown fences
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text).strip()

    # Find outermost JSON object
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    else:
        return {"student_answer": _last_resort_extract(response_text),
                "confidence": 0.3, "notes": "json_parse_failed"}

    # Fix trailing commas
    text = _TRAILING_COMMA.sub(r'\1', text)

    # Fix single-quoted keys (best-effort, avoids touching Arabic)
    if '"student_answer"' not in text and "'student_answer'" in text:
        for key in ("student_answer", "confidence", "notes"):
            text = text.replace(f"'{key}'", f'"{key}"')

    # Parse
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {"student_answer": _last_resort_extract(response_text),
                    "confidence": 0.25, "notes": "json_decode_error"}

    # Unwrap nested JSON inside student_answer
    student_answer = parsed.get("student_answer", "")
    if isinstance(student_answer, dict):
        student_answer = student_answer.get("student_answer", str(student_answer))
    elif isinstance(student_answer, str):
        s = student_answer.strip()
        if s.startswith('{') and 'student_answer' in s:
            try:
                inner = json.loads(s)
                student_answer = inner.get("student_answer", student_answer)
            except json.JSONDecodeError:
                pass

    if student_answer in (None, "null", "None"):
        student_answer = ""
    if student_answer:
        student_answer = student_answer.replace('|', '').strip()

    # Coerce confidence
    try:
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "student_answer": student_answer,
        "confidence": round(confidence, 2),
        "validated": bool(parsed.get("validated", True)),
        "notes": str(parsed.get("notes", "")),
    }


# ============================================
# SECTION 4 — POST-PROCESSING FOR ANSWERS
# Applied after LLM extraction and JSON parse.
# ============================================

_MATH_EXPR        = re.compile(r'[\d\+\-x/=\s]{3,}')
_DOUBLED_OPS      = re.compile(r'([+\-x/=])\1+')
_OP_SPACING       = re.compile(r'\s*([+\-x/=])\s*')
_ARABIC_DIGITS_MAP = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
_PERSIAN_DIGITS_MAP = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')


def _remove_isolated_noise_chars(text: str) -> str:
    """Remove single stray noise characters that appear alone on a line."""
    clean_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if len(stripped) == 1 and not stripped.isalnum() and not re.match(
                f'[{_ARABIC_RANGE}]', stripped):
            continue
        clean_lines.append(line)
    return '\n'.join(clean_lines)


def _fix_math_spacing(text: str) -> str:
    """Normalise spacing around operators in numeric runs; leaves Arabic intact."""
    def replacer(m: re.Match) -> str:
        expr = _OP_SPACING.sub(r' \1 ', m.group(0))
        expr = _DOUBLED_OPS.sub(r'\1', expr)
        expr = re.sub(r' {2,}', ' ', expr)
        return expr.strip()
    return _MATH_EXPR.sub(replacer, text)


def postprocess_student_answer(answer: str) -> str:
    """
    Post-process extracted student answer for accuracy and cleanliness.

    Extends the existing clean_output() with:
      • Double-safety Arabic/Persian digit conversion
      • Removal of bidi/invisible characters
      • Removal of pipe symbols
      • Removal of isolated single-char noise lines
      • Normalised math operator spacing
      • Collapsed doubled operators (== → =)
      • Trimmed trailing Arabic punctuation artefacts
    """
    if not answer:
        return answer

    text = answer

    # Double-safety digit normalisation
    text = text.translate(_ARABIC_DIGITS_MAP)
    text = text.translate(_PERSIAN_DIGITS_MAP)

    # Remove bidi / invisible characters
    for ch in ('\u200b', '\u200c', '\u200d', '\ufeff',
               '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
               '\u2066', '\u2067', '\u2068', '\u2069'):
        text = text.replace(ch, '')

    # Remove pipe and bar symbols
    text = text.replace('|', '').replace('│', '').replace('┃', '')

    # Remove isolated noise characters on their own lines
    text = _remove_isolated_noise_chars(text)

    # Normalise math spacing and fix doubled operators
    text = _fix_math_spacing(text)

    # Strip trailing Arabic commas / punctuation OCR often appends
    text = re.sub(r'[،,;\.]{1,3}\s*$', '', text, flags=re.MULTILINE)

    # Collapse 3+ newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ============================================
# SECTION 5 — CONFIDENCE CALIBRATOR
# Adjusts LLM confidence based on answer quality.
# ============================================

def calibrate_confidence(answer: str, llm_confidence: float) -> float:
    """
    Adjust the LLM's self-reported confidence based on answer quality signals.

    Heuristics applied:
      • Empty answer              → 0.0
      • Very short (≤2 chars)     → −0.15
      • Only punctuation/noise    → −0.20
      • Valid math A op B = C     → +0.05
      • Contains Arabic words     → +0.05
      • Repeated-char noise       → −0.20
    """
    if not answer:
        return 0.0

    score   = llm_confidence
    stripped = answer.strip()

    if len(stripped) <= 2:
        score -= 0.15
    if re.match(r'^[\W_]+$', stripped, re.UNICODE):
        score -= 0.20
    if re.search(r'\d+\s*[+\-x/]\s*\d+\s*=\s*\d+', stripped):
        score += 0.05
    if re.search(f'[{_ARABIC_RANGE}]{{2,}}', stripped):
        score += 0.05
    if re.search(r'(.)\1{4,}', stripped):
        score -= 0.20

    return round(max(0.0, min(1.0, score)), 2)


# ============================================
# SECTION 6 — ORIGINAL CLEANING FUNCTIONS
# Kept verbatim — still used as safety net.
# ============================================

def convert_arabic_to_western_digits(text: str) -> str:
    """Convert Arabic-Indic digits to Western digits"""
    digit_map = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
        '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
        '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
    }
    for arabic, western in digit_map.items():
        text = text.replace(arabic, western)
    return text


def fix_operators(text: str) -> str:
    """Fix common operator misreads and clean up spacing."""
    if not text:
        return text
    return text


def clean_output(text: str) -> str:
    """Final cleaning of output - NO pipes, NO wrong operators"""
    if not text:
        return text

    text = text.replace('|', '')
    text = text.replace('↵', '\n')
    text = re.sub(r'\|+', '', text)

    if text.strip().startswith('{') and 'student_answer' in text:
        try:
            nested = json.loads(text)
            if isinstance(nested, dict) and 'student_answer' in nested:
                text = nested['student_answer']
        except Exception:
            pass

    text = convert_arabic_to_western_digits(text)
    text = fix_operators(text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()
    text = re.sub(r'[│┃┊┋]', '', text)

    return text


# ============================================
# SECTION 7 — ORIGINAL PROMPT (kept as-is)
# ============================================

ANSWER_EXTRACTION_PROMPT = """
You extract student answers from Arabic primary school math exams.

CRITICAL - ARABIC MATH IS WRITTEN RIGHT TO LEFT (RTL):
When Arabic students write math, they write from RIGHT to LEFT.
The result comes FIRST, then the operation, then the numbers.

Examples of Arabic RTL math:
- Student writes: "3380 = 2870 - 6250"
  This means in English: 6250 - 2870 = 3380

- Student writes: "9630 = 5250 + 3380"  
  This means in English: 3380 + 5250 = 9630

EXTRACTION RULES:
1. PRESERVE THE EXACT RTL ORDER the student wrote
2. Use ONLY Western digits: 0 1 2 3 4 5 6 7 8 9
3. Use ONLY these operators: - + x / =
4. NO decimal points: 2089 NOT 20.89
5. NO pipe symbols | anywhere in output
6. NO vertical bars or separators
7. Clean single answer per line

IMPORTANT:
- NO pipe symbols | 
- NO vertical bars
- Return ONLY valid JSON
- student_answer must be a string
"""


def get_prompt(section_type: str) -> str:
    """Return the extraction prompt"""
    return ANSWER_EXTRACTION_PROMPT


# ============================================
# SECTION 8 — HELPER FUNCTIONS (original)
# ============================================

def extract_section_number(filename: str) -> str:
    numbers = re.findall(r'\d+', filename)
    return numbers[0] if numbers else "unknown"


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def clean_json_response(response_text: str) -> str:
    """Extract JSON from response (original — kept as fallback)"""
    if response_text.startswith('```json'):
        response_text = response_text[7:]
    elif response_text.startswith('```'):
        response_text = response_text[3:]
    if response_text.endswith('```'):
        response_text = response_text[:-3]

    start = response_text.find('{')
    end   = response_text.rfind('}')
    if start != -1 and end != -1:
        return response_text[start:end + 1]

    return response_text.strip()


# ============================================
# SECTION 9 — CORE EXTRACTION (patched)
# Only 4 targeted changes inside this function;
# everything else is identical to the original.
# ============================================

def extract_student_answer(image_path: str, section_type: str = "answer_zone") -> dict:
    """Extract student answer with clean output"""
    try:
        encoded_string = encode_image(image_path)

        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{encoded_string}"
            },
            include_image_base64=True
        )

        if len(ocr_response.pages) == 0:
            return {"student_answer": "", "confidence": 0.0}

        raw_text = ocr_response.pages[0].markdown

        # [IMPROVED] Preprocess: strips markdown, bidi chars, normalises Arabic glyphs
        raw_text = preprocess_ocr_text(raw_text)

        if not raw_text.strip():
            return {"student_answer": "", "confidence": 0.0}

        # [IMPROVED] Enhanced system + user prompt replaces the original base_prompt build
        system_prompt, user_prompt = build_enhanced_prompt(section_type, raw_text)

        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=1000
        )

        response_text = chat_response.choices[0].message.content.strip()

        # [IMPROVED] Robust JSON parsing with auto-repair
        parsed         = validate_and_repair_json(response_text)
        student_answer = parsed.get("student_answer", "")
        confidence     = parsed.get("confidence", 0.5)

        if student_answer in [None, "null", "None", ""]:
            student_answer = ""

        if student_answer:
            # [IMPROVED] Enhanced post-processing + confidence calibration
            student_answer = postprocess_student_answer(student_answer)
            # [IMPROVED] Math validation: correct RTL equation order if needed
            student_answer, math_valid, math_notes = validate_math_in_answer(student_answer)
            if math_notes:
                parsed["notes"] = (parsed.get("notes", "") + " | " + math_notes).strip(" |")
            if not math_valid:
                parsed["validated"] = False
            confidence = calibrate_confidence(student_answer, confidence)

        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except Exception:
            confidence = 0.5

        return {
            "student_answer": student_answer,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {"student_answer": "", "confidence": 0.0, "error": str(e)}


# ============================================
# SECTION 10 — BATCH PROCESSING (original)
# ============================================

def process_all_sections(folder_path: str, section_type: str = "answer_zone") -> list:
    folder = Path(folder_path)

    images = []
    for ext in ['.png', '.jpg', '.jpeg']:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))

    if not images:
        print("❌ No images found")
        return []

    unique = {}
    for img in images:
        if img.stem not in unique:
            unique[img.stem] = img

    images = sorted(unique.values(), key=lambda x: extract_section_number(x.name))

    print(f"\n{'='*60}")
    print(f"📁 {folder_path}")
    print(f"📸 {len(images)} images")
    print(f"{'='*60}\n")

    results = []

    for idx, img in enumerate(images, 1):
        section = extract_section_number(img.name)
        print(f"[{idx}/{len(images)}] Section {section}: {img.name}")

        extracted = extract_student_answer(str(img), section_type)

        try:
            section_num = int(section)
        except Exception:
            section_num = section

        result = {
            "section_number": section_num,
            "filename":       img.name,
            "student_answer": extracted.get("student_answer", ""),
            "confidence":     extracted.get("confidence", 0.0),
        }

        results.append(result)

        if result["student_answer"]:
            preview = result["student_answer"][:80].replace('\n', ' | ')
            print(f"  ✍️  {preview}")
            print(f"  🎯 {result['confidence']:.0%}")
        else:
            print("  📭 No answer")
        print()

    return results


# ============================================
# SECTION 11 — SAVE & SUMMARY (original)
# ============================================

def save_results(results: list, folder_path: str, output_folder: str) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    output = {
        "metadata": {
            "source": folder_path,
            "total":  len(results),
            "date":   datetime.now().isoformat(),
            "note":   "Clean output: NO pipe symbols | NO wrong operators",
        },
        "results": results,
    }

    json_file = Path(output_folder) / f"answers_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Saved: {json_file}")
    return json_file


def print_summary(results: list):
    if not results:
        return

    total        = len(results)
    with_answers = sum(1 for r in results if r['student_answer'])

    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"Total:           {total}")
    print(f"With answers:    {with_answers}")
    print(f"Without answers: {total - with_answers}")

    if with_answers > 0:
        avg_conf = sum(r['confidence'] for r in results if r['student_answer']) / with_answers
        print(f"Avg confidence:  {avg_conf:.0%}")

    print(f"\n📝 Sample outputs (clean — no pipes):")
    print("-" * 40)
    for r in results[:3]:
        if r['student_answer']:
            print(f"Section {r['section_number']}: {r['student_answer'][:60]}")


# ============================================
# MAIN
# ============================================

def main():
    print("🚀 Student Answer Extractor - Clean Output")
    print("=" * 50)
    print("✓ NO pipe symbols |")
    print("✓ Clean operators: - + x / =")
    print("✓ Western digits only (0-9)")
    print("✓ Preserves Arabic RTL math order")
    print("=" * 50)

    if not Path(INPUT_FOLDER).exists():
        print(f"\n❌ Folder not found: {INPUT_FOLDER}")
        return

    results = process_all_sections(INPUT_FOLDER)

    if results:
        save_results(results, INPUT_FOLDER, OUTPUT_FOLDER)
        print_summary(results)
        print("\n✅ Complete!")
    else:
        print("\n❌ No results")


if __name__ == "__main__":
    main()