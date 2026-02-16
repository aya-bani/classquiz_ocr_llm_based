from .exam_corrector import ExamCorrector
from .question_extractor import QuestionExtractor
from .agents_config import AgentsConfig

class AgentsManager:
    def __init__(self):
        AgentsConfig.validate()
        self.extractor = QuestionExtractor()  
        self.corrector = ExamCorrector()
    
    def extract_questions(self, folder_path, is_submission, save_results=False, output_path=None):
        """Process exam images - delegates to QuestionExtractor"""
        return self.extractor.process_exam(folder_path, is_submission, save_results, output_path)
    
    def correct_exam(self, exam_content, submission_content):
        """Grade student submission - delegates to ExamCorrector"""
        return self.corrector.correct_exam(exam_content, submission_content)
    