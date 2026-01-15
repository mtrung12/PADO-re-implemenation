from .prompts import explain_prompt_build
from .utils import generate_response

def generate_explaination(ctrait, ctext, model_name, induce='high', prompt_type='pado', max_new_tokens: int = None):    
    sys_p, usr_p = explain_prompt_build(ctrait, ctext, induce=induce, prompt_type=prompt_type)
    return generate_response(sys_p, usr_p, model=model_name, max_new_tokens=max_new_tokens)