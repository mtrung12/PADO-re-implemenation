from .prompts import judgement_prompt_build
from .utils import generate_response
from typing import List, Union
import json
import re

def generate_judgement(ctrait: str, text: Union[str, List[str]], explanation1: Union[str, List[str]], explanation2: Union[str, List[str]], model_name: str, max_new_tokens: int = None, pipeline = None, log_filepath: str = None):
    sys_p, usr_p = judgement_prompt_build(ctrait, text, explanation1, explanation2)
    # All requests are now sent to generate_response, which handles single prompts for API models
    # and batching for HuggingFace models.
    return generate_response(sys_p, usr_p, model=model_name, max_new_tokens=max_new_tokens, pipeline=pipeline, log_filepath=log_filepath)


def extract_judgement_prediction(judgement_response: str) -> str:
    if judgement_response.startswith("Error:"):
        return "Unknown"
    
    lower_response = judgement_response.lower()
    
    try:
        # Find the start index of "final judgement"
        start_index = lower_response.index("final judgement")
        
        # Get text after "final judgement"
        prediction_text = lower_response[start_index + len("final judgement"):]
        
        # Find "high"
        if "high" in prediction_text:
            return "high"
        # Find "low"
        elif "low" in prediction_text:
            return "low"
        # If neither
        else:
            return "unknown"
            
    except ValueError:
        # "final judgement" not found
        return "unknown"