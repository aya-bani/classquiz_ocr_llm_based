from exam_corrector import ExamCorrector

# Example exam content (correct answers)
exam_content = [
    {
        "question_number": 1,
        "question_text": "What is the capital of France?",
        "correct_answer": "Paris",
        "max_score": 1
    },
    {
        "question_number": 2,
        "question_text": "Solve: 2 + 2",
        "correct_answer": "4",
        "max_score": 1
    }
]

# Example student submission
submission_content = [
    {
        "question_number": 1,
        "answer": "Paris"
    },
    {
        "question_number": 2,
        "answer": "5"
    }
]

def main():
    corrector = ExamCorrector()
    result = corrector.correct_exam(exam_content, submission_content)
    print("\nGrading Result:")
    print(result)

if __name__ == "__main__":
    main()
