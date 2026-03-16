"""Type-specific prompts for section-level OCR + LLM extraction."""

QUESTION_EXTRACTION_PROMPT = """
You analyze one exam section image.

Extract:
1) printed question text
2) handwritten student answer written with pen
3) confidence score in [0, 1]

Rules:
- Focus on the student's handwritten response, not printed examples.
- If no handwritten answer exists, set student_answer to null.
- Keep question concise but faithful to image text.
- Return ONLY valid JSON.

JSON schema:
{
  "question": "string or null",
  "student_answer": "string or null",
  "confidence": 0.0
}
""".strip()


HANDWRITTEN_ANSWER_PROMPT = """
You analyze one exam section image.

Priority:
- Detect pen-written student answer with high recall.
- Ignore decorative elements and most printed instructions.

Rules:
- Keep the student's original wording/spelling.
- If handwriting is partially unreadable, include readable part and
  use [illegible] for unclear parts.
- If there is no handwritten answer, set student_answer to null.
- Return ONLY valid JSON.

JSON schema:
{
  "question": "string or null",
  "student_answer": "string or null",
  "confidence": 0.0
}
""".strip()


INSTRUCTION_PROMPT = """
You analyze one exam section image that may contain instructions.

Extract:
- printed instruction/question text if present
- handwritten student answer if any

Rules:
- For pure instruction sections, student_answer is usually null.
- Do not hallucinate missing text.
- Return ONLY valid JSON.

JSON schema:
{
  "question": "string or null",
  "student_answer": "string or null",
  "confidence": 0.0
}
""".strip()


SECTION_TYPE_TO_PROMPT = {
    "question": QUESTION_EXTRACTION_PROMPT,
    "answer_zone": HANDWRITTEN_ANSWER_PROMPT,
    "instruction": INSTRUCTION_PROMPT,
    "unknown": QUESTION_EXTRACTION_PROMPT,
}
