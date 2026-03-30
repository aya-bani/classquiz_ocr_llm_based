# ============================================
# ARABIC RTL MATH PROMPT - CLEAN OUTPUT
# ============================================

ANSWER_EXTRACTION_PROMPT = """
You extract student answers from Arabic primary school math exams.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL - ARABIC MATH IS READ RIGHT TO LEFT (RTL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When Arabic students write math, they write and READ from RIGHT to LEFT.

READING ORDER (RTL):
- The EYE starts at the RIGHT side of the equation
- The RESULT appears FIRST (on the RIGHT)
- Then the OPERATOR
- Then the NUMBERS flow to the LEFT

How to READ Arabic math:
Student writes: "3380 = 2870 - 6250"
READING ORDER (RIGHT TO LEFT):
Step 1: Start at RIGHT → "3380" (result)
Step 2: Move LEFT → "=" (equals)
Step 3: Move LEFT → "2870" (second number)
Step 4: Move LEFT → "-" (minus)
Step 5: Move LEFT → "6250" (first number)
Meaning in English (LTR): 6250 - 2870 = 3380

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RTL CALCULATION PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pattern 1 - Subtraction (RTL): "3380 = 2870 - 6250" → 3380 = 2870 - 6250
Pattern 2 - Addition (RTL): "9630 = 5250 + 3380" → 9630 = 5250 + 3380
Pattern 3 - Multiplication (RTL): "12 = 3 × 4" → 12 = 3 × 4
Pattern 4 - Division (RTL): "5 = 4 ÷ 20" → 5 = 4 ÷ 20

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHILD HANDWRITING (AGES 5-10) - CRITICAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Students are 5-10 years old. Their handwriting has specific characteristics:

1. MESSY/UNEVEN WRITING:
   - Letters may be disconnected or overlapping
   - Numbers may be reversed (3 written as ε, 6 as 9, 7 with a line through it)
   - Size varies dramatically within same answer

2. NUMBER RECOGNITION IN RTL CONTEXT:
   - 0 (zero) may look like a circle or oval
   - 1 (one) may be just a vertical line
   - 4 (four) may be open or closed
   - 7 (seven) may have a horizontal line through the middle
   - Use math context to determine the intended number

3. CROSSED-OUT/CORRECTED ANSWERS:
   - Students often cross out wrong answers
   - Look for the FINAL answer (usually the last one written)
   - If multiple answers visible, take the one that appears to be final

4. ARABIC LETTERS:
   - Connected letters may be disconnected
   - Dots may be missing or misplaced
   - Transcribe the INTENDED word based on context

5. ANSWER LOCATION:
   - Answer is usually BELOW التعليمة
   - May be in a box, on a line, or in blank space
   - Look for the area where the student wrote

6. PARTIAL ANSWERS:
   - Students may write incomplete calculations
   - Example: "6250 - 2870 = " → extract "6250 - 2870 = "
   - In RTL: "= 2870 - 6250" (result missing)

7. NOISE & MARKS:
   - Ignore teacher marks (red pen, checkmarks, X's)
   - Ignore stray pencil marks, eraser smudges

8. CONFIDENCE SCORING:
   - 0.90-1.00: Clear, legible, confident extraction
   - 0.70-0.89: Mostly clear, minor uncertainty
   - 0.50-0.69: Some characters unclear, context helped
   - 0.30-0.49: Hard to read, significant guessing
   - 0.00-0.29: Cannot extract reliably

9. WHEN UNCERTAIN:
   - If a number is ambiguous, use math context to determine
   - If a word is illegible, mark as [illegible]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTRACTION RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PRESERVE THE EXACT RTL ORDER the student wrote
2. Use ONLY Western digits: 0 1 2 3 4 5 6 7 8 9
3. Use ONLY these operators: - + x / =
4. NO decimal points: 2089 NOT 20.89
5. NO pipe symbols | anywhere in output
6. NO vertical bars or separators
7. Clean single answer per line
8. For ambiguous numbers, use math context to determine intended value
9. For multiple answers, take the final (uncrossed) answer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES OF CORRECT OUTPUT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example 1 - Simple RTL equation:
Student writes: 3380 = 2870 - 6250
Output: {"student_answer": "3380 = 2870 - 6250", "confidence": 0.95}

Example 2 - Direct calculation:
Student writes: 6250-2870=3380
Output: {"student_answer": "6250-2870=3380", "confidence": 0.95}

Example 3 - Arabic text with math:
Student writes: ثمن المشتريات=7655
Output: {"student_answer": "ثمن المشتريات=7655", "confidence": 0.90}

Example 4 - Multiple lines:
Student writes:
350
275
50
Output: {"student_answer": "350\n275\n50", "confidence": 0.85}

Example 5 - Clean numbers only:
Student writes: 7655
Output: {"student_answer": "7655", "confidence": 0.95}

Example 6 - Messy number with context:
Student writes messy "3" that looks like "8"
Context: subtraction, result should be 3380
Output: {"student_answer": "3380", "confidence": 0.85}

Example 7 - Crossed out answer:
Student writes "3380 = 2870 - 6250" then crosses it, writes "3380" below
Output: {"student_answer": "3380", "confidence": 0.85}

Example 8 - Partial calculation (RTL):
Student writes: "= 2870 - 6250"
Output: {"student_answer": "= 2870 - 6250", "confidence": 0.70}

IMPORTANT:
- NO pipe symbols | 
- NO vertical bars
- Return ONLY valid JSON
- student_answer must be a string
- READ Arabic math from RIGHT to LEFT
- PRESERVE the RTL order in output
"""

def get_prompt(section_type: str) -> str:
    """Return the extraction prompt"""
    return ANSWER_EXTRACTION_PROMPT

SYSTEM_PROMPT = "Extract student answers. Preserve RTL order. For children's messy handwriting, use context. NO pipe symbols |. Return ONLY valid JSON."