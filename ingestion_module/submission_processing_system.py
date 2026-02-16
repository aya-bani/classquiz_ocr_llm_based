from typing import List, Dict
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from marker_module import MarkerManager
from Layout_module import LayoutManager
from logger_manager import LoggerManager
from agents_module import AgentsManager
from agents_module.exam_corrector import ExamCorrector
from .exams_repository import ExamsRepository


class SubmissionProcessingSystem:
    def __init__(self, max_workers: int = 5):
        """
        Initialize submission processing system
        
        Args:
            max_workers: Number of concurrent workers
            api_requests_per_minute: API rate limit for Gemini
        """
        self.marker_manager = MarkerManager()
        self.layout_manager = LayoutManager()
        self.logger = LoggerManager.get_logger(__name__)
        self.agent_manager = AgentsManager()
        self.exam_corrector = ExamCorrector()
        self.repo = ExamsRepository()
        self.executor = None
        self.max_workers = max_workers
    
    def __enter__(self):
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.agent_manager:
            self.agent_manager.extractor.close()
        self.repo.close()
    
    def process_submission(self, submission_id: int, student_name: str, 
                          student_id: str, pages_submitted: List[Image.Image]) -> Dict:
        """
        Process a complete student submission
        
        Args:
            submission_id: Unique submission identifier
            student_name: Name of the student
            student_id: Student ID number
            pages_submitted: List of scanned page images
            
        Returns:
            Dictionary with processing results and grading
        """
        self.logger.info(f"Processing submission {submission_id} for student {student_name}")
        
        if not pages_submitted:
            self.logger.warning(f"No pages submitted for submission {submission_id}")
            return {"error": "No pages submitted"}
        
        # Step 1: Insert submission into database
        try:
            db_submission_id = self.repo.insert_submission(student_name, student_id)
            self.logger.info(f"Submission registered in DB with ID: {db_submission_id}")
        except Exception as e:
            self.logger.error(f"Failed to register submission: {e}")
            return {"error": f"Failed to register submission: {str(e)}"}
        
        # Step 2: Scan and identify exams in submission
        try:
            marker_result = self.marker_manager.scan_submission(submission_id, pages_submitted)
        except Exception as e:
            self.logger.error(f"Marker failed: {e}")
            return {"error": f"Exam detection failed: {str(e)}"}
        
        if not marker_result:
            self.logger.warning(f"No exams detected in submission {submission_id}")
            return {"error": "No exams detected in submission"}
        
        # Step 3: Process each detected exam
        results = {
            "submission_id": db_submission_id,
            "student_name": student_name,
            "student_id": student_id,
            "exams": {}
        }
        
        for exam in marker_result:
            exam_id = exam['exam_id']
            exam_path = exam['output_path']
            
            try:
                exam_result = self._process_single_exam(
                    exam_id, 
                    db_submission_id, 
                    exam_path
                )
                results["exams"][exam_id] = exam_result
                
            except Exception as e:
                self.logger.error(f"Failed to process exam {exam_id}: {e}", exc_info=True)
                results["exams"][exam_id] = {
                    "error": str(e),
                    "exam_id": exam_id
                }
        
        return results
    
    def _process_single_exam(self, exam_id: int, submission_id: int, 
                            exam_path: Path) -> Dict:
        """
        Process a single exam from submission
        
        Args:
            exam_id: ID of the exam
            submission_id: ID of the submission
            exam_path: Path to the exam PDF
            
        Returns:
            Dictionary with exam processing results including grading
        """
        self.logger.info(f"Processing exam {exam_id} from submission {submission_id}")
        
        # Step 1: Extract sections from submission
        try:
            layout_result = self.layout_manager.process_submission(
                exam_id, 
                submission_id, 
                exam_path
            )
            sections_dir = layout_result.get('sections_dir')
            
            if not sections_dir or not Path(sections_dir).exists():
                raise ValueError(f"Sections directory not found: {sections_dir}")
                
        except Exception as e:
            self.logger.error(f"Layout processing failed: {e}")
            return {"error": f"Layout processing failed: {str(e)}"}
        
        # Step 2: Classify and extract content from submission
        try:
            output_path = Path(f"submission_{submission_id}_exam_{exam_id}_classification.json")
            submission_content = self.agent_manager.extract_questions(
                folder_path=Path(sections_dir),
                is_submission=True,
                save_results=True,
                output_path=output_path
            )
            
            if not submission_content:
                raise ValueError("No content extracted from submission")
                
        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return {
                "error": f"Content extraction failed: {str(e)}",
                "sections_dir": str(sections_dir)
            }
        
        # Step 3: Fetch correct answers from database
        try:
            exam_data = self.repo.get_exam_content(exam_id)
            
            if not exam_data:
                raise ValueError(f"Exam {exam_id} not found in database")
            
            exam_content = exam_data.get('content', [])
            
            if not exam_content:
                raise ValueError(f"No content found for exam {exam_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to fetch exam content: {e}")
            return {
                "error": f"Failed to fetch exam content: {str(e)}",
                "submission_content": submission_content
            }
        
        # Step 4: Grade the submission
        try:
            grading_result = self.exam_corrector.correct_exam(
                exam_content=exam_content,
                submission_content=submission_content
            )
            print("-----------------Grading Result-------------------")
            print(grading_result)

            if "error" in grading_result:
                self.logger.warning(f"Grading had errors: {grading_result['error']}")
            
        except Exception as e:
            self.logger.error(f"Grading failed: {e}")
            grading_result = {
                "error": str(e),
                "total_score": 0,
                "max_score": 0,
                "percentage": 0
            }
        
        # Step 5: Save grading results to database
        try:
            grading_id = self.repo.insert_grading_result(
                submission_id=submission_id,
                exam_id=exam_id,
                grading_data=grading_result
            )
            grading_result["grading_id"] = grading_id
            self.logger.info(f"Grading saved with ID: {grading_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save grading results: {e}")
            grading_result["save_error"] = str(e)
        
        # Return complete result
        return {
            "exam_id": exam_id,
            "exam_data": {
                "level": exam_data.get('level'),
                "subject": exam_data.get('subject')
            },
            "sections_dir": str(sections_dir),
            "submission_content": submission_content,
            "grading": grading_result
        }
    
    def get_submission_results(self, submission_id: int) -> Dict:
        """
        Retrieve all results for a submission
        
        Args:
            submission_id: ID of the submission
            
        Returns:
            Dictionary with all grading results
        """
        try:
            results = self.repo.get_submission_results(submission_id)
            return {
                "submission_id": submission_id,
                "results": results
            }
        except Exception as e:
            self.logger.error(f"Failed to retrieve results: {e}")
            return {
                "submission_id": submission_id,
                "error": str(e)
            }