"""Type-specific prompts for section-level OpenAI vision extraction."""

QUESTION_EXTRACTION_PROMPT = """
You are analyzing a single scanned exam section image.

Tasks:
1. Read the printed question or instruction text.
2. Detect the student's handwritten answer written with pen or pencil.
3. Ignore decorative shapes, page noise, and repeated printed examples.
4. Estimate a confidence score between 0 and 1.

Rules:
- Prioritize real student handwriting over printed content.
- Preserve the student's wording and spelling as written.
- If handwriting is partly unreadable, keep the readable text and use
  [illegible]
  only where necessary.
- If no student answer exists in the section, set student_answer to null.
- Return valid JSON only, with no markdown and no explanation.
""".strip()


HANDWRITTEN_ANSWER_PROMPT = """
You are analyzing an answer zone from a scanned school exam.

Primary goal:
- Extract the student's handwritten answer with high recall.

Secondary goal:
- Capture the related printed question text only if it is clearly visible.

Rules:
- Focus on pen or pencil marks made by the student.
- Ignore printed instructions unless they are needed to understand the answer.
- Do not invent missing words.
- If no handwritten answer is present, set student_answer to null.
- Return valid JSON only, with no markdown and no explanation.
""".strip()


INSTRUCTION_PROMPT = """
You are analyzing an instruction or mixed-content exam section.

Tasks:
- Extract the printed instruction or question text.
- Extract any handwritten student answer if one is present.

Rules:
- Many instruction sections have no student answer; in that case use null.
- Ignore layout artifacts and repeated printed labels.
- Do not hallucinate text that is not visible.
- Return valid JSON only, with no markdown and no explanation.
""".strip()


QUESTION_PROMPT = QUESTION_EXTRACTION_PROMPT
ANSWER_EXTRACTION_PROMPT = HANDWRITTEN_ANSWER_PROMPT


SECTION_TYPE_TO_PROMPT = {
    "question": QUESTION_EXTRACTION_PROMPT,
    "answer_zone": HANDWRITTEN_ANSWER_PROMPT,
    "instruction": INSTRUCTION_PROMPT,
    "unknown": QUESTION_EXTRACTION_PROMPT,
}
