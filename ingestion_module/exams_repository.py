import json
import threading
from contextlib import contextmanager
from typing import List, Dict, Optional
from .ingestion_config import IngestionConfig

class ExamsRepository:
    def __init__(self):
        """Initialize with thread-local storage"""
        IngestionConfig.validate()
        self._local = threading.local()
    
    @contextmanager
    def _get_cursor(self):
        """
        Thread-safe context manager for database operations.
        Automatically handles commit/rollback and cursor cleanup.
        """
        if not hasattr(self._local, 'conn') or self._local.conn.closed:
            self._local.conn = IngestionConfig.get_connection()
        
        cursor = self._local.conn.cursor()
        
        try:
            yield cursor
            self._local.conn.commit()
        except Exception as e:
            self._local.conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def insert_exam(self, level: str, subject: str) -> int:
        """Insert a new exam and return its ID"""
        with self._get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO exams (level, subject)
                VALUES (%s, %s)
                RETURNING exam_id;
                """,
                (level, subject)
            )
            exam_id = cur.fetchone()[0]
            return exam_id
    
    def update_exam(self, exam_id: int, content: list[dict], 
                    blank_pdf_path, correction_pdf_path):
        """Update exam with content and file paths"""
        blank_pdf_path = str(blank_pdf_path)
        correction_pdf_path = str(correction_pdf_path)
        json_strings = [json.dumps(item) for item in content]
        
        with self._get_cursor() as cur:
            cur.execute(
                """
                UPDATE exams
                SET blank_pdf_path = %s,
                    correction_pdf_path = %s,
                    content = %s::json[]
                WHERE exam_id = %s;
                """,
                (blank_pdf_path, correction_pdf_path, json_strings, exam_id)
            )
    
    def get_exam_content(self, exam_id: int) -> Optional[Dict]:
        """
        Fetch exam content by ID
        
        Returns:
            Dictionary with exam details including content, or None if not found
        """
        with self._get_cursor() as cur:
            cur.execute(
                """
                SELECT exam_id, level, subject, content, blank_pdf_path, correction_pdf_path
                FROM exams
                WHERE exam_id = %s;
                """,
                (exam_id,)
            )
            row = cur.fetchone()
            
            if not row:
                return None
            
            # Parse the content from JSON
            content = []
            if row[3]:  # content field
                for json_str in row[3]:
                    if isinstance(json_str, str):
                        content.append(json.loads(json_str))
                    else:
                        content.append(json_str)
            
            return {
                "exam_id": row[0],
                "level": row[1],
                "subject": row[2],
                "content": content,
                "blank_pdf_path": row[4],
                "correction_pdf_path": row[5]
            }
    
    def insert_submission(self, student_name: str, student_id: str) -> int:
        """
        Insert a new submission and return its ID
        
        Args:
            student_name: Name of the student
            student_id: Student identification number
            
        Returns:
            submission_id
        """
        with self._get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO submissions (student_name, student_id, submitted_at)
                VALUES (%s, %s, NOW())
                RETURNING submission_id;
                """,
                (student_name, student_id)
            )
            submission_id = cur.fetchone()[0]
            return submission_id
    
    def insert_grading_result(self, submission_id: int, exam_id: int, 
                            grading_data: Dict) -> int:
        """
        Insert grading results for a submission
        
        Args:
            submission_id: ID of the submission
            exam_id: ID of the exam
            grading_data: Dictionary containing grading results
            
        Returns:
            grading_id
        """
        with self._get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO grading_results (
                    submission_id, 
                    exam_id, 
                    total_score, 
                    max_score,
                    percentage,
                    detailed_results,
                    feedback,
                    graded_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING grading_id;
                """,
                (
                    submission_id,
                    exam_id,
                    grading_data.get('total_score', 0),
                    grading_data.get('max_score', 0),
                    grading_data.get('percentage', 0),
                    json.dumps(grading_data.get('detailed_results', [])),
                    grading_data.get('overall_feedback', '')
                )
            )
            grading_id = cur.fetchone()[0]
            return grading_id
    
    def get_submission_results(self, submission_id: int) -> List[Dict]:
        """
        Get all grading results for a submission
        
        Args:
            submission_id: ID of the submission
            
        Returns:
            List of grading results
        """
        with self._get_cursor() as cur:
            cur.execute(
                """
                SELECT 
                    g.grading_id,
                    g.exam_id,
                    e.level,
                    e.subject,
                    g.total_score,
                    g.max_score,
                    g.percentage,
                    g.detailed_results,
                    g.feedback,
                    g.graded_at
                FROM grading_results g
                JOIN exams e ON g.exam_id = e.exam_id
                WHERE g.submission_id = %s
                ORDER BY g.graded_at DESC;
                """,
                (submission_id,)
            )
            
            results = []
            for row in cur.fetchall():
                detailed_results = json.loads(row[7]) if row[7] else []
                
                results.append({
                    "grading_id": row[0],
                    "exam_id": row[1],
                    "level": row[2],
                    "subject": row[3],
                    "total_score": row[4],
                    "max_score": row[5],
                    "percentage": row[6],
                    "detailed_results": detailed_results,
                    "feedback": row[8],
                    "graded_at": row[9]
                })
            
            return results
    
    def close(self):
        """Close the connection for THIS thread"""
        if hasattr(self._local, 'conn') and not self._local.conn.closed:
            self._local.conn.close()