from .prompts import judgement_prompt_build
from .utils import generate_response

def generate_judgement(ctrait, explanation1, explanation2, use_gpt=False):
    sys_p, usr_p = judgement_prompt_build(ctrait, explanation1, explanation2)
    return generate_response(sys_p, usr_p, use_gpt=use_gpt)

def extract_judgement_prediction(judgement_response: str) -> str:
    for line in judgement_response.splitlines():
        if line.startswith("Final Prediction:"):
            return line.split(":")[1].strip().strip("**")
    return "Unknown"