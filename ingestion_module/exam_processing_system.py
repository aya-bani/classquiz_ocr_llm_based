from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .exams_repository import ExamsRepository
from pdf2image import convert_from_path
from Layout_module import LayoutManager
from marker_module import MarkerManager
from logger_manager import LoggerManager
from agents_module import AgentsManager

class ExamProcessingSystem:
    def __init__(self, max_workers=6):
        """
        Initialize the exam processing system
        
        Args:
            max_workers: Number of concurrent exam processing threads
            api_requests_per_minute: Gemini API rate limit (4=free tier, 1000=paid tier)
        """
        self.marker_manager = MarkerManager()
        self.layout_manager = LayoutManager()
        self.logger = LoggerManager.get_logger(__name__)
        self.executor = None
        self.repo = ExamsRepository()
        self.all_exams = {}
        self.completed_tasks = 0
        self.max_workers = max_workers
        self.agent_manager = AgentsManager()

    def __enter__(self):
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.agent_manager:
            self.agent_manager.extractor.close()
        self.repo.close()

    def add_exams(self, exams_path: Path) -> dict[int, dict]:
        """
        Add all exams from a directory recursively.
        Each exam must have 'blank.pdf' and 'correction.pdf' in the same folder.
        """
        exams_to_process = []
        
        for folder in exams_path.rglob("*"):
            if folder.is_dir():
                blank_path = folder / "blank.pdf"
                corrected_path = folder / "correction.pdf"
                
                if blank_path.exists() and corrected_path.exists():
                    parts = folder.parts
                    subject = parts[-1]
                    level = parts[-2]
                    exam_name = folder.name
                    
                    # Insert exam in DB to get unique ID
                    exam_id = self.repo.insert_exam(level, subject)
                    self.logger.info(f"Registered exam {exam_name} | Level: {level} | Subject: {subject}")

                    # Store info (convert Path to string immediately)
                    self.all_exams[exam_id] = {
                        "blank": str(blank_path),
                        "correction": str(corrected_path),
                        "marked_exam": None,
                        "correction_sections": None
                    }

                    exams_to_process.append((exam_id, corrected_path, blank_path))

        # Submit all exams to be processed
        if exams_to_process:
            return self.process_exams(exams_to_process)
        return {}

    def process_exams(self, exams_data) -> dict[int, dict]:
        """
        Submit all exams to the executor and track progress.
        Each exam has three tasks: marker processing, correction processing, and classification.
        f3 starts as soon as its corresponding f2 completes.
        """
        all_futures = {}

        # Submit tasks for all exams
        for exam_id, corrected_path, blank_path in exams_data:
            f1 = self.executor.submit(self.marker_manager.mark_exam, exam_id, blank_path)
            f2 = self.executor.submit(self.layout_manager.process_correction, exam_id, corrected_path)
            
            # Create closure that captures exam_id correctly
            def make_classifier(eid, bp, cp):
                def submit_classification(f2_future):
                    """Helper function to submit f3 after f2 completes"""
                    try:
                        correction_result = f2_future.result()
                        correction_dir = correction_result.get('sections_dir')
                        
                        if correction_dir and Path(correction_dir).exists():
                            output_path = Path(f"exam_{eid}_classification.json")
                            content = self.agent_manager.extract_questions(
                                folder_path=Path(correction_dir),
                                is_submission=False,
                                save_results=True,
                                output_path=output_path
                            )
                            
                            # Ensure paths are strings
                            blank_str = str(bp)
                            correction_str = str(cp)
                            
                            self.repo.update_exam(eid, content, blank_str, correction_str)
                            return content
                        else:
                            self.logger.warning(f"Correction directory not found for exam {eid}")
                            return None
                    except Exception as e:
                        self.logger.error(f"Failed to process classification for exam {eid}: {e}", exc_info=True)
                        return None
                return submit_classification
            
            # f3 will automatically start when f2 completes
            f3 = self.executor.submit(make_classifier(exam_id, blank_path, corrected_path), f2)
            all_futures[exam_id] = {"f1": f1, "f2": f2, "f3": f3}

        # Track progress
        total_tasks = len(all_futures) * 3
        self.logger.info(f"Total tasks to complete: {total_tasks}")
        
        all_task_futures = [
            future 
            for exam in all_futures.values() 
            for future in [exam["f1"], exam["f2"], exam["f3"]]
        ]
        
        for future in as_completed(all_task_futures):
            self.completed_tasks += 1
            self.logger.info(f"Progress: {self.completed_tasks}/{total_tasks} tasks completed")

        # Collect results per exam
        results = {}
        for exam_id, futures_dict in all_futures.items():
            try:
                marked_result = futures_dict["f1"].result()
                correction_result = futures_dict["f2"].result()
                classification_result = futures_dict["f3"].result()
                
                self.all_exams[exam_id].update({
                    "marked_exam": str(marked_result) if marked_result else None,
                    "correction_sections": str(correction_result) if correction_result else None,
                    "content_classification": classification_result
                })
                
            except Exception as e:
                self.logger.error(f"Processing failed for exam {exam_id}: {e}", exc_info=True)
                
            results[exam_id] = self.all_exams[exam_id]

        return results
    
    def _load_pdf(self, pdf_path: Path) -> tuple:
        """Load PDF file and return list of pages"""
        pages = convert_from_path(pdf_path, dpi=300)
        return pages