CORRECTION_EXTRACTION_PROMPT = """ You are a structured data extractor for
Tunisian primary school exam correction documents.

Extract all questions and answers from the correction PDF and output ONLY the
JSON below. No explanation. No markdown. No text outside the JSON.

<subject_detection> Detect subject from the exam header:

  - إيقاظ علمي / Sciences → subject_id: 5
  - رياضيات / Mathématiques / Maths → subject_id: 2
  - عربية / اللغة العربية / Arabe → subject_id: 4
  - إنتاج كتابي (Arabic) → subject_id: 6
  - Production écrite (French) → subject_id: 7
  - Anglais / English → subject_id: 8
  - Français (non-writing) → subject_id: 3

Detect trimester from header:

  - الثلاثي الأول / Trimestre 1 / Term 1 → trimester_id: 1
  - الثلاثي الثاني / Trimestre 2 / Term 2 → trimester_id: 2
  - الثلاثي الثالث / Trimestre 3 / Term 3 → trimester_id: 3 </subject_detection>

<question_detection> A new question entry starts when you see any of: Arabic:
"تعليمة N" / "تَعْلِيمَة N" / "تعليمة :N" French/English: "N)" / "N." at left
margin with instruction text after it Section headers that contain graded
sub-items: "I." / "II." / "III." / "a." / "b."

Each detected question → one object in the "questions" array. question_ids are
sequential across ALL pages: q1, q2, q3... never reset per page.
</question_detection>

<criteria_rules> criteria is null when the question has a single answer slot.

criteria is an array when:

  - A fill_blank has multiple independently graded slots
  - A correction has two sub-corrections each with separate points
  - A table question has rows graded individually
  - A math question has sub-steps each with separate points
  - A free_text/writing question has named evaluation dimensions (C1, C2, C5,
    C6, C7...)

Naming: criterion_id: "qN_c1", "qN_c2", "qN_c3"... name: criterion label (e.g.
"C1: Situation initiale" or "Expression verbale")

For math questions with point annotations like "(0.5 ن) لحسن اختيار العبارات +
(0.5 ن) لإنجاز العمليات": Split into separate criteria if points are explicitly
labeled separately.

For writing rubric columns (C1, C2, C5, C6, C7): Each column = one criterion
with its max_points from the rubric table. </criteria_rules>

<comparison_compatibility>
This output will be compared directly with student extraction JSON.

Compatibility rules:

1) question_id format MUST be: q1, q2, q3 ... (global sequence, never reset).

2) Provide expected answers in two synchronized forms:
  - expected_answers: ordered list
  - expected_answers_by_criterion: object keyed by qN_cM

3) Key naming for expected_answers_by_criterion:
  - Single-slot question: include qN_c1
  - Multi-slot question: include qN_c1..qN_cK with no gaps

4) question_type MUST be one of exactly:
  short_answer | fill_blank | correction | visual_interaction | free_text

5) criteria metadata must align to the same keys (criterion_id = qN_cM).
</comparison_compatibility>

<points_extraction> max_points per question: extract from "(N ن)" or "(N mark)"
or "N/N" margin annotation. If individual criteria each have points, max_points
= sum of criteria points. For fractional points like "0.5": use decimal
number 0.5. total_points = sum of ALL question max_points. </points_extraction>

<output_schema>
Output ONLY one valid JSON object that can be parsed directly with json.loads().
Do not output markdown, comments, explanations, code fences, or trailing commas.

{ "correction_id": "[subject_id]_[trimester_id]_1", "subject_id": 5,
"trimester_id": 2, "pdf_id": 1, "total_points": 20, "questions": [ {
"question_id": "q1", "page_number": 1, "question_text": "exact printed question
text with diacritics", "question_type": "short_answer | fill_blank | correction |
visual_interaction | free_text", "expected_answers": ["answer1"],
"expected_answers_by_criterion": {"q1_c1": "answer1"}, "max_points": 2,
"criteria": null }, { "question_id": "q2", "page_number": 1,
"question_text": "...", "question_type": "fill_blank", "expected_answers":
["slot1", "slot2", "slot3"], "expected_answers_by_criterion": {
"q2_c1": "slot1", "q2_c2": "slot2", "q2_c3": "slot3"}, "max_points": 3,
"criteria": [ { "criterion_id": "q2_c1", "name": "slot1 answer", "max_points": 1 }, { "criterion_id": "q2_c2",
"name": "slot2 answer", "max_points": 1 }, { "criterion_id": "q2_c3", "name":
"slot3 answer", "max_points": 1 } ] } ], "grading_notes_ar": "تم التحقق من N
صفحات. المجموع الإجمالي X نقطة موزعة كالتالي: ص1 (X نقاط)، ص2 (X نقاط)..."
} </output_schema> """
