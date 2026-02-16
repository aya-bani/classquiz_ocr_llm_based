import time
from typing import Dict, List
from logger_manager import LoggerManager
from google.api_core import exceptions
import json
from .prompts import GRADING_PROMPT
from .agent import Agent

class ExamCorrector:
    """
    Grades student submissions by comparing with correct answers
    """
    
    def __init__(self):
        """
        Initialize the exam corrector
        """
        self.logger = LoggerManager.get_logger(__name__)
    
    def correct_exam(self, exam_content: List[Dict], 
                    submission_content: List[Dict]) -> Dict:
        """
        Grade a student's submission against the correct answers
        
        Args:
            exam_content: List of questions with correct answers from DB
            submission_content: List of student's answers
            
        Returns:
            Dictionary with grading results and feedback
        """
        self.logger.info("Starting exam correction")
        
        # Build the grading prompt
        prompt = self._build_grading_prompt(exam_content, submission_content)
        attempt = 0
        while True:
            try:
                Agent.wait_if_needed()
                print("-----------------Grading Prompt-------------------")
                print(prompt)
                response = Agent.get_model().generate_content(prompt)
                print("-----------------Grading Response-------------------")
                print(response.text)
                grading_result = self._parse_grading_response(response.text)
                
                self.logger.info(
                    f"Grading complete: {grading_result.get('total_score', 0)}/"
                    f"{grading_result.get('max_score', 0)} "
                    f"({grading_result.get('percentage', 0):.1f}%)"
                )
                
                return grading_result
                
            except exceptions.ResourceExhausted as e:
                retry_delay = self._parse_retry_delay(str(e))
                attempt += 1
                self.logger.warning(
                    f"Rate limit hit, waiting {retry_delay:.1f}s "
                    f"(attempt {attempt})"
                )
                Agent.handle_rate_limit(retry_delay)
  
            except Exception as e:
                self.logger.error(f"Grading error: {e}", exc_info=True)
                return self._create_error_result(str(e))
        
    
    def _build_grading_prompt(self, exam_content: List[Dict], 
                                submission_content: List[Dict]) -> str:
            """
            Build the prompt for grading
            
            Args:
                exam_content: Correct answers from exam
                submission_content: Student's submission
                
            Returns:
                Formatted prompt string
            """
            try:
                print("i'm here************************************************************************************************")
                prompt = GRADING_PROMPT.format(
                    exam_content=exam_content,
                    submission_content=submission_content
                )
                return prompt
            except KeyError as e:
                self.logger.error(f"Missing key in GRADING_PROMPT template: {e}")
                raise ValueError(f"Invalid GRADING_PROMPT template - missing placeholder: {e}")
            except Exception as e:
                self.logger.error(f"Failed to build grading prompt: {e}", exc_info=True)
                raise ValueError(f"Failed to build grading prompt: {str(e)}")
    
    def _parse_grading_response(self, response_text: str) -> Dict:
        """
        Parse the grading response from AI
        
        Args:
            response_text: Raw response from model
            
        Returns:
            Parsed grading result
        """
        # Remove markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        try:
            result = json.loads(text)
            
            # Validate required fields
            required_fields = ['detailed_results', 'total_score', 'max_score', 'percentage']
            for field in required_fields:
                if field not in result:
                    self.logger.warning(f"Missing field in grading result: {field}")
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}\nResponse: {text}")
            return self._create_error_result(f"Failed to parse grading: {str(e)}")
    
    def _parse_retry_delay(self, error_str: str) -> float:
        """Extract retry delay from error message"""
        retry_delay = 60
        if "retry in" in error_str.lower():
            try:
                import re
                match = re.search(r'retry in ([\d.]+)s', error_str, re.IGNORECASE)
                if match:
                    retry_delay = float(match.group(1))
            except:
                pass
        return retry_delay
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create an error result structure"""
        return {
            "error": error_message,
            "detailed_results": [],
            "total_score": 0,
            "max_score": 0,
            "percentage": 0,
            "overall_feedback": f"Grading failed: {error_message}",
            "strengths": [],
            "areas_for_improvement": [],
            "grade": "N/A"
        }