import time
import threading
from .agents_config import AgentsConfig
import google.generativeai as genai


class Agent:
    """Thread-safe rate limiter for API calls"""
    
    @classmethod
    def get_model(cls):
        """Get Gemini model name and API key from environment (singleton)"""
        if not hasattr(cls, "_model_instance_"):
            genai.configure(api_key=AgentsConfig.GEMINI_API_KEY)
            cls._model_instance_ = genai.GenerativeModel(AgentsConfig.GEMINI_MODEL_NAME)
        return cls._model_instance_

    @classmethod
    def wait_if_needed(cls):
        """Wait if necessary to respect rate limits"""
        if not hasattr(cls, "_lock_"):
            cls._lock_ = threading.Lock()
            cls.last_request_time = 0
        
        with cls._lock_:
            current_time = time.time()
            time_since_last = current_time - cls.last_request_time
            if time_since_last < AgentsConfig.MIN_INTERVAL:
                wait_time = AgentsConfig.MIN_INTERVAL - time_since_last
                time.sleep(wait_time)
            cls.last_request_time = time.time()
    
    @classmethod
    def handle_rate_limit(cls, retry_delay: float):
        """
        Handle rate limit with coordinated waiting across threads.
        Only the first thread to arrive sleeps the full duration.
        Subsequent threads check if enough time has passed.
        
        Args:
            retry_delay: Delay in seconds from the API error message
        """
        if not hasattr(cls, "_lock_"):
            cls._lock_ = threading.Lock()
            cls.last_request_time = 0
        
        with cls._lock_:
            current_time = time.time()
            
            # Calculate when we can make the next request
            # Using last_request_time as the baseline
            next_allowed_time = cls.last_request_time + retry_delay + 1
            
            # Check if we need to wait
            if current_time < next_allowed_time:
                wait_time = next_allowed_time - current_time
                time.sleep(wait_time)
            
            # Update last request time to now
            cls.last_request_time = time.time()