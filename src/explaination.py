from .prompts import explain_prompt_build
from .utils import generate_response

def generate_explaination(ctrait, ctext, induce='high', prompt_type='pado', use_gpt=False):    
    sys_p, usr_p = explain_prompt_build(ctrait, ctext, induce=induce, prompt_type=prompt_type)

    return generate_response(sys_p, usr_p)