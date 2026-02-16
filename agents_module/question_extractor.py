import time
from concurrent.futures import ThreadPoolExecutor
from .agents_config import AgentsConfig
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from logger_manager import LoggerManager
from google.api_core import exceptions
from .prompts import CLASSIFICATION_PROMPT
from .agent import Agent



class QuestionExtractor:
    """
    Classifies exam questions and extracts structured content using Google Gemini
    """

    def __init__(self):
        """
        Initialize the classifier with Google Gemini API
        """
        self.logger = LoggerManager.get_logger(__name__)
        # Initialize executor
        self.max_workers = 5
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def __enter__(self):
        """Context manager entry - executor already initialized in __init__"""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Shutdown the executor"""
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def close(self):
        """Shutdown the executor"""
        if self.executor:
            self.executor.shutdown(wait=True)


    def process_exam(
        self, 
        folder_path: Path,
        is_submission: bool = False, 
        save_results: bool = False,
        output_path: Optional[Path] = None
    ) -> List[Dict]:
        """
        Process multiple question images in batch
        
        Args:
            folder_path: Path to folder containing question images
            is_submission: Whether this is a submission or correction image
            save_results: Whether to save results to JSON file
            output_path: Path to save results (if save_results=True)
            
        Returns:
            List of classification and extraction results sorted by section number
        """
        # Check if executor is initialized
        if self.executor is None:
            raise RuntimeError(
                "Executor not initialized. This should not happen if using __init__ properly."
            )
        
        # Check if folder exists and is a directory
        if not folder_path.exists():
            self.logger.error(f"Folder does not exist: {folder_path}")
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")
        
        if not folder_path.is_dir():
            self.logger.error(f"Path is not a directory: {folder_path}")
            raise NotADirectoryError(f"Path is not a directory: {folder_path}")
        
        image_paths = [
            p
            for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        ]
        
        total = len(image_paths)
        
        # Warn if no images found
        if total == 0:
            self.logger.warning(f"No image files found in {folder_path}")
            return []
        
        self.logger.info(f"Starting batch processing of {total} images")
        
        # Submit all tasks and collect futures with their corresponding image paths
        futures = []
        for image_path in image_paths:
            future = self.executor.submit(self.process_image, image_path, is_submission)
            futures.append((future, image_path))
        
        # Wait for all tasks to complete and collect results
        results = []
        for future, image_path in futures:
            try:
                result = future.result()  # Blocks until this specific task completes
                results.append(result)
            except Exception as e:
                self.logger.error(f"Task failed: {e}")
                results.append({
                    "meta_data": {
                        "image_path": str(image_path),
                        "image_name": image_path.name
                    },
                    "error": str(e),
                    "question_type": "ERROR",
                    "confidence": 0.0
                })
        
        # Sort results by section number
        results.sort(
            key=lambda r: self._extract_section_number(
                r.get("meta_data", {}).get("image_name", "")
            )
        )
        
        # Calculate statistics
        successful = sum(1 for r in results if "error" not in r)
        self.logger.info(
            f"Batch processing complete: {successful}/{total} successful"
        )
        
        # Save results if requested
        if save_results:
            if output_path is None:
                output_path = Path("question_analysis_results.json")
            
            self._save_results(results, output_path)
        
        return results


    def process_image(self, image_path: Path, is_submission: bool) -> Dict:
        """
        Process a single question image
        
        Args:
            image_path: Path to the question image
            is_submission: Whether this is a submission or correction image
            
        Returns:
            Dictionary containing classification and extraction results
        """
        self.logger.info(f"Processing image: {image_path}")
        
        try:
            with Image.open(image_path) as image:
                # Step 1: Classify question type
                question_type, confidence = self._classify_question_type(image)
                
                # Step 2: Get extraction prompt for the question type
                extraction_prompt = AgentsConfig.get_extraction_prompt(question_type, is_submission)
                
                if extraction_prompt is None:
                    self.logger.warning(f"No extraction prompt found for question type: {question_type}")
                    return {
                        "question_type": question_type,
                        "confidence": confidence,
                        "content": {
                            "error": "No extraction prompt available for this question type"
                        },
                        "meta_data": {
                            "image_path": str(image_path),
                            "image_name": image_path.name
                        }
                    }
                
                # Step 3: Extract content
                content = self._extract_content(image, extraction_prompt)
            
            return {
                "question_type": question_type,
                "confidence": confidence,
                "content": content,
                "meta_data": {
                    "image_path": str(image_path),
                    "image_name": image_path.name
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}", exc_info=True)
            return {
                "meta_data": {
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                },
                "error": str(e),
                "question_type": "ERROR",
                "confidence": 0.0
            }
    

    
    def _classify_question_type(self, image: Image.Image) -> Tuple[str, float]:
        """
        Classify the type of question in the image with retry logic
        
        Args:
            image: PIL Image of the question
            
        Returns:
            Tuple of (question_type, confidence_score)
        """
        attempt = 0
        while True:
            try:
                # Thread-safe rate limiting
                Agent.wait_if_needed()
                
                # Get classification from Gemini
                response = Agent.get_model().generate_content([
                    CLASSIFICATION_PROMPT, 
                    image
                ])
                result = self._parse_json_response(response.text)
                
                # Extract classification results
                question_type = result.get("question_type", "UNKNOWN").upper()
                confidence = float(result.get("confidence", 0.5))
                
                self.logger.info(
                    f"Classified as {question_type} with confidence {confidence:.2f}"
                )
                
                return question_type, confidence
                
            except exceptions.ResourceExhausted as e:
                # Extract retry delay from error
                retry_delay = self._parse_retry_delay(str(e))
                attempt += 1
                
                self.logger.warning(
                    f"Rate limit hit, waiting {retry_delay:.1f}s before retry "
                    f"(attempt {attempt})"
                )
                Agent.handle_rate_limit(retry_delay)

                
            except Exception as e:
                self.logger.error(f"Classification error: {e}", exc_info=True)
                return "UNKNOWN", 0.0


    def _extract_content(self, image: Image.Image, extraction_prompt: str) -> Dict:
        """
        Extract structured content with retry logic
        
        Args:
            image: PIL Image of the question
            extraction_prompt: Prompt template for content extraction
            
        Returns:
            Structured content dictionary
        """
        attempt = 0
        while True:
            try:
                # Thread-safe rate limiting
                Agent.wait_if_needed()
                
                response = Agent.get_model().generate_content([extraction_prompt, image])
                content = self._parse_json_response(response.text)
                
                if not content or content.get("error"):
                    self.logger.warning(f"Content extraction returned error or empty result")
                
                return content
                
            except exceptions.ResourceExhausted as e:
                retry_delay = self._parse_retry_delay(str(e))
                attempt += 1
                
                self.logger.warning(
                    f"Rate limit hit during extraction, waiting {retry_delay:.1f}s "
                    f"(attempt {attempt})"
                )
                Agent.handle_rate_limit(retry_delay)
                
            except Exception as e:
                self.logger.error(f"Content extraction error: {e}", exc_info=True)
                return {
                    "error": str(e),
                    "raw_text": "Failed to extract content"
                }

    def _parse_retry_delay(self, error_str: str) -> float:
        """Extract retry delay from error message"""
        retry_delay = 60  # Default
        if "retry in" in error_str.lower():
            try:
                import re
                match = re.search(r'retry in ([\d.]+)s', error_str, re.IGNORECASE)
                if match:
                    retry_delay = float(match.group(1))
            except:
                pass
        return retry_delay
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """
        Parse JSON from model response, handling markdown code blocks
        
        Args:
            response_text: Raw response text from model
            
        Returns:
            Parsed JSON dictionary
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
            return json.loads(text)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}\nResponse: {text}")
            return {
                "raw_text": text, 
                "error": f"Failed to parse JSON: {str(e)}"
            }
    
    def _extract_section_number(self, image_name: str) -> int:
        """
        Extract section number from image filename
        
        Args:
            image_name: Image filename in format "exam_{exam_id}_section_{section_number}.ext"
            
        Returns:
            Section number as integer, or -1 if parsing fails
        """
        try:
            import re
            # Match pattern: exam_{exam_id}_section_{section_number}
            match = re.search(r'section[_\s-](\d+)', image_name, re.IGNORECASE)
            if match:
                return int(match.group(1))
        except Exception as e:
            self.logger.warning(f"Failed to extract section number from '{image_name}': {e}")
        
        return -1  # Return -1 for items that can't be parsed (they'll be sorted last)
      
    def _save_results(self, results: List[Dict], output_path: Path) -> None:
        """
        Save results to JSON file
        
        Args:
            results: List of processing results
            output_path: Path to save the JSON file
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}", exc_info=True)
    
    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        Generate statistics from batch processing results
        
        Args:
            results: List of processing results
            
        Returns:
            Dictionary containing statistics
        """
        if not results:
            return {"error": "No results to analyze"}
        
        total = len(results)
        successful = sum(1 for r in results if "error" not in r)
        failed = total - successful
        
        # Count question types
        type_counts = {}
        confidence_sum = 0
        confidence_count = 0
        
        for result in results:
            if "error" not in result:
                q_type = result.get("question_type", "UNKNOWN")
                type_counts[q_type] = type_counts.get(q_type, 0) + 1
                
                conf = result.get("confidence", 0)
                confidence_sum += conf
                confidence_count += 1
        
        avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0
        
        return {
            "total_processed": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "average_confidence": avg_confidence,
            "question_type_distribution": type_counts,
            "most_common_type": (max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None)
        }
    
