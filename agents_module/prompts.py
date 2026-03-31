# ============================================================================
# GRADING PROMPT
# ============================================================================
GRADING_PROMPT = """
    You are an expert teacher grading a student's exam. 
    You will be given the correct answers (exam_content) and the student's answers (submission_content).

    Your task is to:
    1. Compare each student answer with the correct answer
    2. Award points based on correctness (partial credit allowed)
    3. Provide specific feedback for each question
    4. Calculate total score and percentage
    5. Give overall feedback and suggestions for improvement

    **IMPORTANT**: Respond ONLY with valid JSON, no markdown formatting, no extra text.

    Exam Content (Correct Answers):
    {exam_content}

    Student Submission (Student's Answers):
    {submission_content}

    Respond with a JSON object in this exact format:
    {{
    "detailed_results": [
        {{
        "question_number": 1,
        "question_type": "MCQ",
        "max_points": 10,
        "points_awarded": 8,
        "is_correct": false,
        "feedback": "Your answer is partially correct. You identified the main concept but missed...",
        "student_answer_summary": "Brief summary of what student wrote",
        "correct_answer_summary": "Brief summary of correct answer"
        }}
    ],
    "total_score": 85,
    "max_score": 100,
    "percentage": 85.0,
    "overall_feedback": "Good work overall. Strong understanding of...",
    "strengths": ["Clear explanations", "Good problem-solving"],
    "areas_for_improvement": ["Need more detail in...", "Review the concept of..."],
    "grade": "B+"
    }}

    CRITICAL: Return ONLY the JSON object, nothing else. No markdown, no explanation, just pure JSON.
    """

# ============================================================================
# CLASSIFICATION PROMPT
# ============================================================================
CLASSIFICATION_PROMPT = """
    Analyze this exam question image and classify it into ONE category.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    QUESTION TYPES (choose exactly ONE):

    1. ENONCE
    • General instructions, context, or statement
    • Usually at the beginning of an exam section
    • Provides background or scenario for following questions
    • May include "Read the following...", "Context:", "Instructions:"

    2. WRITING
    • Essay or long-form written response
    • Requires paragraph(s) or extended answer
    • May include "Write about...", "Discuss...", "Explain in detail..."
    • Often has word count requirements

    3. RELATING
    • Matching or correspondence questions
    • Connects items from two lists/columns
    • "Match the following...", "Connect...", "Pair..."
    • Usually has numbered items and lettered options

    4. TABLE
    • Questions involving tables
    • Fill in table cells, analyze table data, complete table
    • Clear table structure with rows and columns
    • May ask to complete missing cells or analyze data

    5. MULTIPLE_CHOICE
    • Questions with 2+ labeled options
    • Options labeled with A/B/C/D or 1/2/3/4
    • "Choose the correct answer", "Select one:"
    • Clear distinct options to choose from

    6. TRUE_FALSE
    • Statements to be marked as true or false
    • "True or False:", "T/F:", "Vrai/Faux:"
    • May have multiple statements to evaluate
    • Binary choice format

    7. FILL_BLANK
    • Sentences or text with missing words
    • Contains blanks: "____", "[___]", "........"
    • "Fill in the blanks", "Complete the sentence"
    • May include word bank

    8. SHORT_ANSWER
    • Brief written response (1-3 sentences)
    • Requires specific factual answer
    • "Define...", "What is...", "Name...", "List..."
    • NOT essay-length

    9. CALCULATION
    • Mathematical or computational problems
    • Contains numbers, formulas, equations
    • "Calculate...", "Solve...", "Find the value..."
    • Requires numerical answer

    10. DIAGRAM
        • Questions with diagrams, graphs, charts, or illustrations
        • May ask to label, analyze, or interpret visual elements
        • "Label the diagram", "Identify parts", "Analyze the graph"
        • Visual component is central to the question

    11. UNKNOWN
        • Cannot reliably determine the type
        • Image quality too poor
        • Question format unclear or unusual
        • Use ONLY as last resort

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    CLASSIFICATION CRITERIA:

    Consider these factors in order of priority:
    1. Primary question format and structure
    2. Layout and visual organization
    3. Presence of specific elements (options, tables, blanks, diagrams)
    4. Question wording and language indicators
    5. Expected answer format

    Decision rules:
    • If question has lettered options (A/B/C/D) → MULTIPLE_CHOICE
    • If question has two columns to connect → RELATING
    • If question has blank spaces in text → FILL_BLANK
    • If question has table structure → TABLE
    • If question asks for paragraph response → WRITING
    • If question has diagram/graph → DIAGRAM (even if also asking calculation)
    • If question is purely T/F → TRUE_FALSE
    • If question needs math calculation → CALCULATION
    • If question asks brief factual answer → SHORT_ANSWER
    • If text provides context for other questions → ENONCE

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    OUTPUT FORMAT:

    Respond ONLY with a valid JSON object (no markdown, no extra text):

    {{
        "question_type": "TYPE_NAME",
        "confidence": 0.95,
        "reasoning": "Brief explanation: why this type was chosen and key indicators observed"
    }}

    IMPORTANT:
    • question_type must be EXACTLY one of the 11 types above in UPPERCASE
    • confidence must be a number between 0.0 and 1.0
    • reasoning should be 1-2 sentences explaining the classification
    • Be decisive: choose the BEST matching type even if not perfect
    • Use UNKNOWN only if truly impossible to classify
    """

# ============================================================================
# BASE EXTRACTION PROMPT - SUBMISSION MODE
# ============================================================================
BASE_PROMPT_SUBMISSION_EXTRACTION = """
    Your task is to extract TWO elements from this student SUBMISSION image.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. CONTENT (Required - Extract the original printed question)
    
    What to extract:
    • The printed/original question text (NOT student's handwritten answers)
    • All question components as specified in EXTRACTION_STRUCTURE
    • Question numbers, point values, and instructions
    
    Rules:
    • Follow EXACTLY the EXTRACTION_STRUCTURE provided below
    • Extract only the fields specified - do not add or remove fields
    • Use null for fields that cannot be found or are illegible
    • Do NOT infer, guess, or add information not visible
    • Preserve exact wording and formatting
    • This is the PRINTED question, not the student's work

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    2. STUDENT_ANSWER (Required - Extract student's handwritten answers)
    
    Look for these indicators:
    ✓ HANDWRITTEN text (written by student)
    ✓ Circled or checked options in multiple choice
    ✓ Filled-in blanks with handwriting
    ✓ Written calculations, work shown, formulas
    ✓ Diagrams labeled by student
    ✓ Crossed-out text (still extract it, mark as crossed out)
    
    Rules:
    • Use the SAME structure as EXTRACTION_STRUCTURE
    • Extract student's exact writing - preserve their wording/spelling even if wrong
    • For illegible handwriting, use "[illegible]"
    • For crossed-out work, use "[crossed out: text]"
    • If student left blank/empty, use null for that field
    • Include partial attempts and corrections
    • Be lenient with messy handwriting - do your best
    
    What to extract by question type:
    • Multiple Choice: Which option(s) student circled/checked
    • Fill Blank: What student wrote in each blank
    • Calculation: Student's work, steps, and final answer
    • Short Answer: Student's written response
    • Table: What student filled in each cell
    • Diagram: What student labeled/drew
    • True/False: Which answers student marked
    • Relating: Student's matches/connections

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    CRITICAL RULES:
    ✓ This is a STUDENT SUBMISSION - extract handwritten work
    ✓ content (printed) and student_answer (handwritten) use identical structure
    ✓ Do NOT look for color markings - this hasn't been graded yet
    ✓ Preserve student's exact answers even if incorrect
    ✓ Extract all visible work including crossed-out attempts
    ✓ Be conservative: null for blank answers, [illegible] for unreadable text
    ✓ Respond ONLY with valid JSON, no extra text or markdown

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    EXTRACTION_STRUCTURE:
    {structure_placeholder}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Required JSON output format:
    {{
        "content": {{}},
        "student_answer": {{}},
        "confidence": 0.95
    }}

    CONFIDENCE SCORING:
    - 0.95-1.0  → Perfect: All text clear, handwriting legible, complete extraction
    - 0.85-0.94 → Excellent: Complete extraction, 1-2 unclear handwritten words
    - 0.70-0.84 → Good: Mostly complete, some illegible handwriting
    - 0.50-0.69 → Fair: Partial extraction, significant illegible sections
    - 0.30-0.49 → Poor: Major illegibility, unclear what student wrote
    - 0.00-0.29 → Failed: Severe quality problems, cannot extract reliably

    HANDWRITING TIPS:
    - Look at context clues for unclear letters
    - Numbers are usually easier than words
    - Capital letters are clearer than lowercase
    - Compare similar letters across the page
    - When truly unsure, mark as [illegible] rather than guess
    """

# ============================================================================
# BASE EXTRACTION PROMPT - CORRECTION MODE
# ============================================================================
BASE_PROMPT_CORRECTION_EXTRACTION = """
    Your task is to extract THREE elements from this exam correction image.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. CONTENT (Required - Extract the original question)
    
    What to extract:
    • The printed/original question text (NOT student answers)
    • All question components as specified in EXTRACTION_STRUCTURE
    • Question numbers, point values, and instructions
    
    Rules:
    • Follow EXACTLY the EXTRACTION_STRUCTURE provided below
    • Extract only the fields specified - do not add or remove fields
    • Use null for fields that cannot be found or are illegible
    • Do NOT infer, guess, or add information not visible
    • Preserve exact wording and formatting

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    2. CORRECT_ANSWER (Optional - Extract marked correct answers)
    
    Look for these visual indicators:
    ✓ GREEN text, highlighting, underlines, or boxes
    ✓ Checkmarks (✓, ✔, ☑) or "correct" labels
    ✓ Text with "Answer:", "Solution:", "Correct:" prefix
    ✓ Circled or boxed text in green/positive marking
    ✓ Text in answer key sections
    
    Rules:
    • Use the SAME structure as EXTRACTION_STRUCTURE
    • Only populate fields that contain the answer (use null for others)
    • If multiple answers are marked, extract all of them
    • If text color is ambiguous, prioritize context clues
    • If NO clear marking exists, return null for entire object
    • Be conservative: when in doubt, return null

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    3. NOTES (Optional - Extract grading/evaluation notes)
    
    Look for these visual indicators:
    ✓ RED text, highlighting, or annotations
    ✓ Point values: "2 pts", "1 point each", "-0.5", "0/3"
    ✓ Grading criteria or rubrics
    ✓ Teacher comments or correction instructions
    ✓ "Note:", "Grading:", "Evaluation:", "Rubric:" prefix
    
    What to extract:
    • Grading rubrics and point distributions
    • Evaluation criteria or requirements
    • Common mistakes or clarifications
    • Instructions for partial credit
    
    Rules:
    • Preserve exact original wording
    • Can be a string or array of strings
    • Include point distributions (e.g., "1 point per correct match")
    • If NO notes exist, return null

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    CRITICAL RULES:
    ✓ content and correct_answer MUST use identical structure (same keys)
    ✓ Extract all three elements independently - NEVER merge them
    ✓ Do NOT infer colors or hidden meanings
    ✓ Ignore student's handwritten answers or crossed-out text
    ✓ Be conservative: null is better than incorrect data
    ✓ Respond ONLY with valid JSON, no extra text or markdown

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    EXTRACTION_STRUCTURE:
    {structure_placeholder}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Required JSON output format:
    {{
        "content": {{}},
        "correct_answer": {{}},
        "notes": null,
        "confidence": 0.95
    }}

    CONFIDENCE SCORING:
    • 0.95-1.0  → Perfect: All fields clear, colors obvious, no ambiguity
    • 0.85-0.94 → Excellent: Complete extraction, 1-2 minor unclear fields
    • 0.70-0.84 → Good: Mostly complete, some ambiguous colors/text
    • 0.50-0.69 → Fair: Partial extraction, significant missing data
    • 0.30-0.49 → Poor: Major issues, illegible text, unclear structure
    • 0.00-0.29 → Failed: Severe quality problems, cannot extract reliably

    SPECIAL CASES:
    • Handwritten over printed → Extract printed text only in 'content'
    • Multiple markings → Extract all clearly marked answers
    • No color but checkmarks → Use checkmarks as answer indicators
    • Partial illegibility → Extract readable parts, use null for rest
    • Mixed languages → Extract in original language, preserve accents
    """

# ============================================================================
# EXTRACTION TEMPLATES - CORRECTION MODE
# ============================================================================
ENONCE_TEMPLATE = """
    Extract enoncé/statement/instructions information.

    STRUCTURE:
    {{
        "title": "heading or title of the section",
        "instructions": [
            "instruction sentence 1",
            "instruction sentence 2"
        ],
        "context": "background information or scenario text"
    }}

    GUIDELINES:
    • title: Usually bold or larger text at the top (may be null if no title)
    • instructions: Separate each instruction as array item
    • context: Paragraph providing background or scenario
    • Use null for any field that is not present
    • Preserve exact wording and formatting indicators
    • Extract complete text content

    NOTE: Enoncé sections typically don't have correct_answer field since they are instructional text
    """

WRITING_TEMPLATE = """
    Extract essay/writing question information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "prompt": "the complete essay question or prompt",
        "requirements": [
            "word limit: 200-250 words",
            "must include personal examples",
            "use formal language"
        ],
        "word_limit": "200-250 words"
    }}

    GUIDELINES:
    • prompt: Extract complete question with all parts
    • requirements: Array of specific requirements (structure, content, style)
    • word_limit: Minimum/maximum word count if specified
    • Include any topics that must be covered
    • Use null for missing fields

    FOR CORRECTIONS (correct_answer field):
    • Extract key points or model answer if provided in green
    • Example: {{"key_points": ["point 1", "point 2"], "sample_answer": "model answer text"}}
    """

RELATING_TEMPLATE = """
    Extract matching/relating question information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "instructions": "how to match items or connect elements",
        "items": [
            {{"id": "1", "text": "First item to match"}},
            {{"id": "2", "text": "Second item to match"}},
            {{"id": "3", "text": "Third item to match"}}
        ],
        "options": [
            {{"id": "A", "text": "First option"}},
            {{"id": "B", "text": "Second option"}},
            {{"id": "C", "text": "Third option"}}
        ]
    }}

    GUIDELINES:
    • items: Usually left column or numbered list (1, 2, 3...)
    • options: Usually right column or lettered list (A, B, C...)
    • Extract in order of appearance (top to bottom)
    • Preserve exact wording for both items and options
    • Include ALL visible items and options
    • Use "id" field for identifiers (numbers for items, letters for options)
    • Use null for illegible items

    FOR CORRECTIONS (correct_answer field):
    • Extract matching pairs showing correct connections
    • Example: {{"matches": [{{"item_id": "1", "option_id": "B"}}, {{"item_id": "2", "option_id": "A"}}, {{"item_id": "3", "option_id": "C"}}]}}
    • Only include matches that are marked in green or with checkmarks
    """

TABLE_TEMPLATE = """
    Extract table-based question information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "question_text": "question or instruction about the table",
        "headers": ["Column 1", "Column 2", "Column 3"],
        "rows": [
            ["row1_col1", "............", "row1_col3"],
            ["row2_col1", "row2_col2", "............"],
            ["row3_col1", "............", "row3_col3"]
        ]
    }}

    GUIDELINES:
    • Extract ALL column headers exactly as shown
    • Extract ALL rows in order from top to bottom
    • For CONTENT: Use "............" (12 dots) for empty cells that students must fill
    • For CORRECT_ANSWER: Replace "............" with the actual answers marked in green
    • Preserve table structure precisely - same number of rows and columns
    • Include row headers if present (first column)
    • Note: rows is 2D array where rows[i][j] is row i, column j

    CONTENT EXAMPLE:
    {{
        "question_text": "Complete the table",
        "headers": ["Country", "Capital", "Population"],
        "rows": [
            ["France", "............", "67 million"],
            ["Germany", "Berlin", "............"],
            ["Italy", "............", "60 million"]
        ]
    }}

    CORRECT_ANSWER EXAMPLE:
    {{
        "question_text": null,
        "headers": ["Country", "Capital", "Population"],
        "rows": [
            ["France", "Paris", "67 million"],
            ["Germany", "Berlin", "83 million"],
            ["Italy", "Rome", "60 million"]
        ]
    }}
    """

MULTIPLE_CHOICE_TEMPLATE = """
    Extract multiple choice question information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "question_text": "the main question or prompt",
        "options": [
            {{"id": "A", "text": "First option text"}},
            {{"id": "B", "text": "Second option text"}},
            {{"id": "C", "text": "Third option text"}},
            {{"id": "D", "text": "Fourth option text"}}
        ]
    }}

    GUIDELINES:
    • question_text: Extract question before the options
    • options: Extract ALL options in order (may be 2, 3, 4, 5+ options)
    • Use "id" field for option identifier (A, B, C, D or 1, 2, 3, 4)
    • Include any sub-text or clarifications within each option
    • Extract multi-line options completely
    • Preserve exact option identifiers as shown in the image

    FOR CORRECTIONS (correct_answer field):
    • Only include the option(s) marked as correct in green or with checkmarks
    • Example: {{"question_text": null, "options": [{{"id": "B", "text": "Second option text"}}]}}
    • If multiple answers are correct, include all marked options
    """

TRUE_FALSE_TEMPLATE = """
    Extract true/false question information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "statements": [
            {{"id": "1", "text": "First statement to evaluate"}},
            {{"id": "2", "text": "Second statement to evaluate"}},
            {{"id": "3", "text": "Third statement to evaluate"}}
        ]
    }}

    GUIDELINES:
    • Extract each statement exactly as written
    • Number statements in order (1, 2, 3...)
    • May be single statement or multiple statements
    • Look for "T/F:", "True/False:", "Vrai/Faux:", "V/F:"
    • Preserve exact statement wording

    FOR CORRECTIONS (correct_answer field):
    • Indicate true/false for each statement marked in green
    • Example: {{"statements": [{{"id": "1", "answer": "true"}}, {{"id": "2", "answer": "false"}}, {{"id": "3", "answer": "true"}}]}}
    • Use lowercase "true" or "false" for answers
    """

FILL_BLANK_TEMPLATE = """
    Extract fill-in-the-blank question information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "text_with_blanks": "The sentence with ............. representing each blank",
        "blank_count": 3
    }}

    GUIDELINES:
    • Use "............." (13 dots) to represent each blank in text_with_blanks
    • Count total number of blanks and put in blank_count
    • Extract complete sentences/paragraphs with blanks marked
    • Preserve text exactly, replacing blank spaces with dots
    • If word bank exists, ignore it (not needed in structure)

    CONTENT EXAMPLE:
    {{
        "question_number": "5",
        "text_with_blanks": "The ............. is the largest planet in our ............. system. It has ............. moons.",
        "blank_count": 3
    }}

    FOR CORRECTIONS (correct_answer field):
    • Provide the complete text with blanks filled in
    • Example: {{"completed_text": "The Jupiter is the largest planet in our solar system. It has 79 moons."}}
    • OR as array: {{"answers": ["Jupiter", "solar", "79"]}}
    """

SHORT_ANSWER_TEMPLATE = """
    Extract short answer question information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "question_text": "the main question",
        "sub_questions": [
            {{"id": "a", "text": "first sub-question"}},
            {{"id": "b", "text": "second sub-question"}}
        ]
    }}

    GUIDELINES:
    • question_text: Main question if no sub-parts
    • sub_questions: Array of sub-parts (a, b, c...) if question is divided
    • If no sub-questions exist, use null for that field
    • Use "id" field for sub-question identifier (a, b, c...)
    • Extract any hints or constraints given in the question

    FOR CORRECTIONS (correct_answer field):
    • Provide sample answer or key points marked in green
    • For single question: {{"answer": "Brief answer text"}}
    • For multiple sub-questions: {{"sub_questions": [{{"id": "a", "answer": "answer to a"}}, {{"id": "b", "answer": "answer to b"}}]}}
    • OR as key points: {{"key_points": ["point 1", "point 2", "point 3"]}}
    """

CALCULATION_TEMPLATE = """
    Extract calculation/math problem information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "problem_statement": "description of the problem",
        "given_values": {{
            "velocity": "10 m/s",
            "time": "5 seconds",
            "distance": "............"
        }},
        "question": "what needs to be calculated"
    }}

    GUIDELINES:
    • problem_statement: Full problem description/scenario
    • given_values: ALL values mentioned - use "............" for unknown values
    • Format: {{"variable_name": "value with unit"}} or {{"variable_name": "............"}}
    • question: Specifically what to calculate/find
    • Preserve mathematical notation and symbols
    • Include constants if provided (e.g., g = 9.8 m/s²)

    CONTENT EXAMPLE:
    {{
        "problem_statement": "A car travels at constant velocity",
        "given_values": {{
            "velocity": "10 m/s",
            "time": "5 seconds",
            "distance": "............"
        }},
        "question": "Calculate the distance traveled"
    }}

    FOR CORRECTIONS (correct_answer field):
    • Provide final result and calculation steps marked in green
    • Example: {{"result": "50 meters", "steps": ["d = v × t", "d = 10 × 5", "d = 50 m"]}}
    • OR simple: {{"answer": "50 meters", "formula_used": "d = v × t"}}
    """

DIAGRAM_TEMPLATE = """
    Extract diagram/illustration-based question information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "question_text": "the question about the diagram",
        "diagram_description": "brief description of what the diagram shows",
        "parts_to_label": [
            {{"id": "A", "description": "arrow pointing to top left"}},
            {{"id": "B", "description": "arrow pointing to center"}},
            {{"id": "C", "description": "arrow pointing to bottom right"}}
        ]
    }}

    GUIDELINES:
    • question_text: What the question asks about the diagram
    • diagram_description: Brief description of diagram type and content
    • parts_to_label: List of parts/arrows that need to be identified
    • Use "id" for label identifiers (A, B, C... or 1, 2, 3...)
    • description: Where the label/arrow is pointing or what it indicates

    CONTENT EXAMPLE:
    {{
        "question_text": "Label the parts of the heart",
        "diagram_description": "anatomical diagram of human heart with 4 arrows",
        "parts_to_label": [
            {{"id": "A", "description": "upper right chamber"}},
            {{"id": "B", "description": "upper left chamber"}},
            {{"id": "C", "description": "lower right chamber"}},
            {{"id": "D", "description": "lower left chamber"}}
        ]
    }}

    FOR CORRECTIONS (correct_answer field):
    • Provide correct labels for each part marked in green
    • Example: {{"labels": [{{"id": "A", "answer": "right atrium"}}, {{"id": "B", "answer": "left atrium"}}, {{"id": "C", "answer": "right ventricle"}}, {{"id": "D", "answer": "left ventricle"}}]}}
    """

GENERIC_TEMPLATE = """
    Extract question content in flexible format.

    STRUCTURE:
    {{
        "question_number": "question number",
        "question_text": "the complete question text",
        "sub_parts": [
            {{"id": "a", "text": "first part of question"}},
            {{"id": "b", "text": "second part of question"}}
        ],
        "additional_info": "any special instructions, figures, or constraints"
    }}

    GUIDELINES:
    • Extract ALL visible text content
    • Preserve structure (numbering, lettering, indentation)
    • sub_parts: Use if question has multiple parts (a, b, c...)
    • additional_info: Note non-text elements (diagrams, tables, images)
    • Maintain original formatting and organization
    • Be thorough - capture all text in the image
    • Use null for missing fields

    FOR CORRECTIONS (correct_answer field):
    • Use flexible format matching content structure
    • Example: {{"answer": "complete answer text"}}
    • OR: {{"sub_parts": [{{"id": "a", "answer": "answer to a"}}, {{"id": "b", "answer": "answer to b"}}]}}
    """

# ============================================================================
# TEMPLATE MAPPING - CORRECTION MODE
# ============================================================================
TEMPLATES_CORRECTION_PROMPT = {
    'ENONCE': ENONCE_TEMPLATE,
    'WRITING': WRITING_TEMPLATE,
    'RELATING': RELATING_TEMPLATE,
    'TABLE': TABLE_TEMPLATE,
    'MULTIPLE_CHOICE': MULTIPLE_CHOICE_TEMPLATE,
    'TRUE_FALSE': TRUE_FALSE_TEMPLATE,
    'FILL_BLANK': FILL_BLANK_TEMPLATE,
    'SHORT_ANSWER': SHORT_ANSWER_TEMPLATE,
    'CALCULATION': CALCULATION_TEMPLATE,
    'DIAGRAM': DIAGRAM_TEMPLATE,
    'UNKNOWN': GENERIC_TEMPLATE
}


# ============================================================================
# EXTRACTION TEMPLATES - SUBMISSION MODE
# ============================================================================
ENONCE_TEMPLATE_SUBMISSION = """
    Extract enoncé/statement/instructions information.

    STRUCTURE:
    {{
        "title": "heading or title of the section",
        "instructions": [
            "instruction sentence 1",
            "instruction sentence 2"
        ],
        "context": "background information or scenario text"
    }}

    FOR SUBMISSION (student_answer field):
    - Enoncé sections typically have no student answers (return null)
    - If student wrote any notes or marks, extract them
    """

WRITING_TEMPLATE_SUBMISSION = """
    Extract essay/writing question information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "prompt": "the complete essay question or prompt",
        "requirements": [
            "word limit: 200-250 words",
            "must include personal examples",
            "use formal language"
        ],
        "word_limit": "200-250 words"
    }}

    FOR SUBMISSION (student_answer field):
    - Extract student's handwritten essay/paragraph
    - Preserve their exact wording and structure
    - Example: {{"essay_text": "Student's complete written response here..."}}
    - If left blank: {{"essay_text": null}}
    - If partially written: {{"essay_text": "[student wrote]: Their partial text here"}}
    """

RELATING_TEMPLATE_SUBMISSION = """
    Extract matching/relating question information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "instructions": "how to match items or connect elements",
        "items": [
            {{"id": "1", "text": "First item to match"}},
            {{"id": "2", "text": "Second item to match"}},
            {{"id": "3", "text": "Third item to match"}}
        ],
        "options": [
            {{"id": "A", "text": "First option"}},
            {{"id": "B", "text": "Second option"}},
            {{"id": "C", "text": "Third option"}}
        ]
    }}

FOR SUBMISSION (student_answer field):
- Extract what student matched/connected (lines, written pairs, etc.)
- Example: {{"matches": [{{"item_id": "1", "student_matched": "C"}}, {{"item_id": "2", "student_matched": "A"}}]}}
- If student wrote "1-C, 2-A, 3-B": extract those pairs
- If no matches made: {{"matches": []}}
"""

TABLE_TEMPLATE_SUBMISSION = """
    Extract table-based question information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "question_text": "question or instruction about the table",
        "headers": ["Column 1", "Column 2", "Column 3"],
        "rows": [
            ["row1_col1", "............", "row1_col3"],
            ["row2_col1", "row2_col2", "............"],
            ["row3_col1", "............", "row3_col3"]
        ]
    }}

    FOR SUBMISSION (student_answer field):
    - Extract what student filled in the table (handwritten entries)
    - Replace "............" with student's handwriting
    - Example: {{"rows": [["France", "paris", "67 million"]]}}  // Note: preserve student's spelling
    - If cell left blank: use null
    - If illegible: use "[illegible]"
    """

MULTIPLE_CHOICE_TEMPLATE_SUBMISSION = """
    Extract multiple choice question information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "question_text": "the main question or prompt",
        "options": [
            {{"id": "A", "text": "First option text"}},
            {{"id": "B", "text": "Second option text"}},
            {{"id": "C", "text": "Third option text"}},
            {{"id": "D", "text": "Fourth option text"}}
        ]
    }}

    FOR SUBMISSION (student_answer field):
    - Extract which option(s) student selected (circled, checked, marked)
    - Look for: circles, checkmarks, X marks, underlining
    - Example: {{"selected_options": [{{"id": "A"}}]}}
    - If multiple selections: include all
    - If no selection: {{"selected_options": []}}
    """

TRUE_FALSE_TEMPLATE_SUBMISSION = """
    Extract true/false question information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "statements": [
            {{"id": "1", "text": "First statement to evaluate"}},
            {{"id": "2", "text": "Second statement to evaluate"}},
            {{"id": "3", "text": "Third statement to evaluate"}}
        ]
    }}

    FOR SUBMISSION (student_answer field):
    - Extract student's true/false marks for each statement
    - Look for: T/F written, checkmarks, circles
    - Example: {{"statements": [{{"id": "1", "student_answer": "true"}}, {{"id": "2", "student_answer": "false"}}]}}
    - If blank: use null
    - Use lowercase "true" or "false"
    """

FILL_BLANK_TEMPLATE_SUBMISSION = """
    Extract fill-in-the-blank question information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "text_with_blanks": "The sentence with ............. representing each blank",
        "blank_count": 3
    }}

    FOR SUBMISSION (student_answer field):
    - Extract what student wrote in each blank
    - Preserve exact spelling even if wrong
    - Example: {{"answers": ["jupter", "solar", "70"]}}  // Note: student's misspelling preserved
    - If blank left empty: use null
    - Example: {{"answers": ["Jupiter", null, "79"]}}  // Second blank empty
    """

SHORT_ANSWER_TEMPLATE_SUBMISSION = """
    Extract short answer question information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "question_text": "the main question",
        "sub_questions": [
            {{"id": "a", "text": "first sub-question"}},
            {{"id": "b", "text": "second sub-question"}}
        ]
    }}

    FOR SUBMISSION (student_answer field):
    - Extract student's handwritten response
    - For single: {{"answer": "Student's written answer here"}}
    - For multiple: {{"sub_questions": [{{"id": "a", "answer": "student wrote this"}}]}}
    - Preserve student's exact words
    - If illegible: {{"answer": "[illegible]"}}
    """

CALCULATION_TEMPLATE_SUBMISSION = """
    Extract calculation/math problem information.

    STRUCTURE:
    {{
        "question_number": "question number",
        "problem_statement": "description of the problem",
        "given_values": {{
            "velocity": "10 m/s",
            "time": "5 seconds",
            "distance": "............"
        }},
        "question": "what needs to be calculated"
    }}

    FOR SUBMISSION (student_answer field):
    - Extract student's work, calculations, and final answer
    - Example: {{"work_shown": ["d = v + t", "d = 10 + 5", "d = 15 m"], "final_answer": "15 meters"}}
    - Include even if wrong formula/answer
    - If crossed out: {{"work_shown": "[illegible]"}}
    """    

DIAGRAM_TEMPLATE_SUBMISSION = """
    Extract diagram/illustration-based question information.
    STRUCTURE:
    {{
        "question_number": "question number",
        "question_text": "the question about the diagram",
        "diagram_description": "brief description of what the diagram shows",
        "parts_to_label": [
            {{"id": "A", "description": "arrow pointing to top left"}},
            {{"id": "B", "description": "arrow pointing to center"}},
            {{"id": "C", "description": "arrow pointing to bottom right"}}
        ]
    }}
    FOR SUBMISSION (student_answer field):
    - Extract student's labels or annotations on the diagram
    - Example: {{"labels": [{{"id": "A", "student_label": "right atrium"}}, {{"id": "B", "student_label": "left atrium"}}]}}
    - Preserve student's exact wording
    - If blank: {{"labels": [{{"id": "A", "student_label": null}}]}}
    """

GENERIC_TEMPLATE_SUBMISSION = """
    Extract question content in flexible format.
    STRUCTURE:
    {{
        "question_number": "question number",
        "question_text": "the complete question text",
        "sub_parts": [
            {{"id": "a", "text": "first part of question"}},
            {{"id": "b", "text": "second part of question"}}
        ],
        "additional_info": "any special instructions, figures, or constraints"
    }}
    FOR SUBMISSION (student_answer field):
    - Extract student's complete response
    - Example: {{"answer": "Student's complete answer text here..."}}
    - If multiple parts: {{"sub_parts": [{{"id": "a", "student_answer": "answer to a"}}, {{"id": "b", "student_answer": "answer to b"}}]}}
    - Preserve student's exact wording and formatting
    """

# ============================================================================
# TEMPLATE MAPPING - SUBMISSION MODE
# ============================================================================
TEMPLATES_SUBMISSIONS_PROMPT = {
    'ENONCE': ENONCE_TEMPLATE_SUBMISSION,
    'WRITING': WRITING_TEMPLATE_SUBMISSION,
    'RELATING': RELATING_TEMPLATE_SUBMISSION,
    'TABLE': TABLE_TEMPLATE_SUBMISSION,
    'MULTIPLE_CHOICE': MULTIPLE_CHOICE_TEMPLATE_SUBMISSION,
    'TRUE_FALSE': TRUE_FALSE_TEMPLATE_SUBMISSION,
    'FILL_BLANK': FILL_BLANK_TEMPLATE_SUBMISSION,
    'SHORT_ANSWER': SHORT_ANSWER_TEMPLATE_SUBMISSION,
    'CALCULATION': CALCULATION_TEMPLATE_SUBMISSION,
    'DIAGRAM': DIAGRAM_TEMPLATE_SUBMISSION,
    'UNKNOWN': GENERIC_TEMPLATE_SUBMISSION
}
