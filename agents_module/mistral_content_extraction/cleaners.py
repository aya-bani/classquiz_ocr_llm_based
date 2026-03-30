import json
import re

def convert_arabic_to_western_digits(text: str) -> str:
    """Convert Arabic-Indic digits to Western digits"""
    digit_map = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
        '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
        '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
    }
    for arabic, western in digit_map.items():
        text = text.replace(arabic, western)
    return text

def fix_operators(text: str) -> str:
    """
    Fix common operator misreads:
    - Sometimes '=' is misread as '+' or vice versa
    - Clean up spacing
    """
    if not text:
        return text
    
    # Replace common misreads (if needed)
    # text = text.replace('+', '=')  # Only if needed
    
    return text

def clean_output(text: str) -> str:
    """Final cleaning of output - NO pipes, NO wrong operators"""
    if not text:
        return text
    
    # Remove pipe symbols
    text = text.replace('|', '')
    
    # Remove any '↵' symbols
    text = text.replace('↵', '\n')
    
    # Remove multiple pipes if any remain
    text = re.sub(r'\|+', '', text)
    
    # Remove nested JSON if present
    if text.strip().startswith('{') and 'student_answer' in text:
        try:
            nested = json.loads(text)
            if isinstance(nested, dict) and 'student_answer' in nested:
                text = nested['student_answer']
        except:
            pass
    
    # Clean numbers
    text = convert_arabic_to_western_digits(text)
    
    # Fix operators
    text = fix_operators(text)
    
    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove any remaining separator symbols
    text = re.sub(r'[│┃┊┋]', '', text)
    
    return text

def clean_json_response(response_text: str) -> str:
    """Extract JSON from response"""
    # Remove markdown
    if response_text.startswith('```json'):
        response_text = response_text[7:]
    elif response_text.startswith('```'):
        response_text = response_text[3:]
    
    if response_text.endswith('```'):
        response_text = response_text[:-3]
    
    # Find JSON object
    start = response_text.find('{')
    end = response_text.rfind('}')
    
    if start != -1 and end != -1:
        return response_text[start:end+1]
    
    return response_text.strip()