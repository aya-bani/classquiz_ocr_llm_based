import json
import base64
from mistralai.client import Mistral

from config import api_key, OCR_MODEL, LLM_MODEL, TEMPERATURE, MAX_TOKENS
from prompts import get_prompt, SYSTEM_PROMPT
from cleaners import clean_json_response, clean_output

client = Mistral(api_key=api_key)

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def extract_student_answer(image_path: str, section_type: str = "answer_zone") -> dict:
    """Extract student answer with clean output"""
    try:
        encoded_string = encode_image(image_path)
        
        ocr_response = client.ocr.process(
            model=OCR_MODEL,
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{encoded_string}"
            },
            include_image_base64=True
        )
        
        if len(ocr_response.pages) == 0:
            return {"student_answer": "", "confidence": 0.0}
        
        raw_text = ocr_response.pages[0].markdown
        
        if not raw_text.strip():
            return {"student_answer": "", "confidence": 0.0}
        
        # Get prompt
        base_prompt = get_prompt(section_type)
        
        user_prompt = f"""
{base_prompt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OCR TEXT FROM IMAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{raw_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTRACT THE STUDENT'S HANDWRITTEN ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Extract EXACTLY what the student wrote.
Preserve the RIGHT-TO-LEFT order.
For ambiguous handwriting, use context to determine the intended answer.
NO pipe symbols | in output.
Return ONLY valid JSON.

Student Answer JSON:
"""

        chat_response = client.chat.complete(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        response_text = chat_response.choices[0].message.content.strip()
        clean_json = clean_json_response(response_text)
        
        # Parse JSON
        try:
            parsed = json.loads(clean_json)
            student_answer = parsed.get("student_answer", "")
            confidence = parsed.get("confidence", 0.5)
        except json.JSONDecodeError:
            student_answer = clean_json
            confidence = 0.5
        
        # Clean up
        if student_answer in [None, "null", "None", ""]:
            student_answer = ""
        
        # Clean the output (removes pipes, fixes operators)
        if student_answer:
            student_answer = clean_output(student_answer)
        
        # Ensure confidence is a float
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except:
            confidence = 0.5
        
        return {
            "student_answer": student_answer,
            "confidence": round(confidence, 2)
        }
        
    except Exception as e:
        return {"student_answer": "", "confidence": 0.0, "error": str(e)}