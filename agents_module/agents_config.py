from dotenv import load_dotenv
import os
from logger_manager import LoggerManager
from .prompts import (
    TEMPLATES_SUBMISSIONS_PROMPT, 
    TEMPLATES_CORRECTION_PROMPT, 
    BASE_PROMPT_SUBMISSION_EXTRACTION, 
    BASE_PROMPT_CORRECTION_EXTRACTION, 
    GENERIC_TEMPLATE, 
    GENERIC_TEMPLATE_SUBMISSION
)

# Load environment once at module level
load_dotenv()

class AgentsConfig:
    """Configuration for agents module with refined prompts"""
    
    logger = LoggerManager.get_logger(__name__)
    # Environment variables
    RATE_LIMIT = int(os.getenv("GEMINI_API_RATE_LIMIT", "150"))
    MIN_INTERVAL = 60.0 / RATE_LIMIT
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    GEMINI_API_KEY = os.getenv("GEMINI_AI_API_KEY")
    
    # Prompt templates
    TEMPLATES_SUBMISSIONS = TEMPLATES_SUBMISSIONS_PROMPT
    TEMPLATES_CORRECTION = TEMPLATES_CORRECTION_PROMPT
    BASE_PROMPT_SUBMISSION = BASE_PROMPT_SUBMISSION_EXTRACTION
    BASE_PROMPT_CORRECTION = BASE_PROMPT_CORRECTION_EXTRACTION
    GENERIC_TEMPLATE_SUB = GENERIC_TEMPLATE_SUBMISSION
    GENERIC_TEMPLATE_CORR = GENERIC_TEMPLATE
    

    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        
        if not cls.GEMINI_API_KEY:
            cls.logger.error("GEMINI_API_KEY not found in environment variables")
            raise ValueError("GEMINI_API_KEY is required")
        
        if not cls.GEMINI_MODEL_NAME:
            cls.logger.warning("GEMINI_MODEL_NAME not set, using default")
        
        cls.logger.info(
            f"Agents config validated - Model: {cls.GEMINI_MODEL_NAME}, "
            f"Rate: {cls.RATE_LIMIT} req/min ({cls.MIN_INTERVAL:.2f}s interval)"
        )
        return True
    
    @classmethod
    def get_extraction_prompt(cls, question_type: str, is_submission: bool) -> str:
        """
        Get complete extraction prompt for OCR
        
        Args:
            question_type: Type of question (uppercase)
            is_submission: Whether this is a submission or correction
            
        Returns:
            Complete prompt with base + structure
        """
        if is_submission:
            base_prompt = cls.BASE_PROMPT_SUBMISSION
            structure = cls.TEMPLATES_SUBMISSIONS.get(
                question_type.upper(),
                cls.GENERIC_TEMPLATE_SUB
            )
        else:
            base_prompt = cls.BASE_PROMPT_CORRECTION
            structure = cls.TEMPLATES_CORRECTION.get(
                question_type.upper(),
                cls.GENERIC_TEMPLATE_CORR
            )

        return base_prompt.format(structure_placeholder=structure)
    
    @classmethod
    def get_classification_template(cls, question_type: str, is_submission: bool) -> str:
        """
        Get classification template for a question type
        
        Args:
            question_type: Type of question (uppercase)
            is_submission: Whether this is a submission or correction
            
        Returns:
            Template string for that question type
        """
        if is_submission:
            return cls.TEMPLATES_SUBMISSIONS.get(
                question_type.upper(),
                cls.GENERIC_TEMPLATE_SUB
            )
        
        return cls.TEMPLATES_CORRECTION.get(
            question_type.upper(),
            cls.GENERIC_TEMPLATE_CORR
        )

    @classmethod
    def get_all_question_types(cls) -> list:
        """Get list of all supported question types"""
        return list(cls.TEMPLATES_CORRECTION.keys())

    @classmethod
    def validate_correction_extraction_result(cls, result: dict) -> tuple:
        """
        Validate extraction result structure
        
        Args:
            result: Extraction result dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required keys
        required_keys = ['content', 'correct_answer', 'notes', 'confidence']
        for key in required_keys:
            if key not in result:
                errors.append(f"Missing required key: {key}")
        
        # Validate confidence
        if 'confidence' in result:
            try:
                conf = float(result['confidence'])
                if not 0 <= conf <= 1:
                    errors.append(f"Confidence {conf} not in range [0, 1]")
            except (TypeError, ValueError):
                errors.append(f"Confidence must be number, got: {type(result['confidence'])}")
        
        # Check content exists
        if result.get('content') is None:
            errors.append("content cannot be null")
        
        # Validate structure match if both content and correct_answer exist
        if result.get('content') and result.get('correct_answer'):
            if isinstance(result['content'], dict) and isinstance(result['correct_answer'], dict):
                content_keys = set(result['content'].keys())
                answer_keys = set(result['correct_answer'].keys())
                
                if content_keys != answer_keys:
                    errors.append(
                        f"Structure mismatch: content keys {content_keys} != "
                        f"correct_answer keys {answer_keys}"
                    )
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            cls.logger.warning(f"Validation failed with {len(errors)} errors: {errors}")
        
        return is_valid, errors