import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_AI_API_KEY"))
try:
    models = list(genai.list_models())
    print(f"Total models found: {len(models)}")
    for m in models:
        print(m.name, m.supported_generation_methods)
except Exception as e:
    print(f"Error: {e}")