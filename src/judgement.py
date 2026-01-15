from .prompts import judgement_prompt_build
from .utils import generate_response

def generate_judgement(ctrait, text, explanation1, explanation2, model_name, max_new_tokens: int = None):
    sys_p, usr_p = judgement_prompt_build(ctrait, text, explanation1, explanation2)
    return generate_response(sys_p, usr_p, model=model_name, max_new_tokens=max_new_tokens)

def extract_judgement_prediction(judgement_response: str) -> str:
    for line in judgement_response.splitlines():
        if line.startswith("Final Prediction:"):
            return line.split(":")[1].strip().strip("**")
    return "Unknown"