import logging
import sys
from pathlib import Path

class LoggerManager:
    _loggers = {}  # Cache to prevent duplicate handlers
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        # Return cached logger if it exists
        if name in LoggerManager._loggers:
            return LoggerManager._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler (UTF-8)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Extract module name for log file
        # e.g., "agents_module.question_extractor" → "agents_module"
        module_name = name.split('.')[0] if '.' in name else name
        log_file = logs_dir / f"{module_name}.log"
        
        # File handler (UTF-8) - module-specific
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Only add handlers if not already present
        if not logger.handlers:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        # Cache the logger
        LoggerManager._loggers[name] = logger
        
        return logger