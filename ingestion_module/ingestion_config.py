from logger_manager import LoggerManager
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

class IngestionConfig:
    """Configuration for ingestion module"""
    
    logger = LoggerManager.get_logger(__name__)
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    

    
    @classmethod
    def validate(cls):
        """Validate that all required config is present"""
        required = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT']
        missing = [key for key in required if not getattr(cls, key)]
        if missing:
            cls.logger.error(f"Missing database config: {missing}")
            raise ValueError(f"Missing database config: {missing}")
        cls.logger.info("Database configuration validated")
        return True
    

    @classmethod
    def get_connection(cls):
        """Return database connection"""
        return psycopg2.connect(
            dbname=cls.DB_NAME,
            user=cls.DB_USER,
            password=cls.DB_PASSWORD,
            host=cls.DB_HOST,
            port=cls.DB_PORT
        )