from .prompts import explain_prompt_build
from .utils import generate_response
import json
import re
from typing import List, Union

def generate_explaination(ctrait, ctext: Union[str, List[str]], model_name: str, induce='high', prompt_type='pado', max_new_tokens: int = None, pipeline = None, log_filepath: str = None):    
    sys_p, usr_p = explain_prompt_build(ctrait, ctext, induce=induce, prompt_type=prompt_type)
    # All requests are now sent to generate_response, which handles single prompts for API models
    # and batching for HuggingFace models.
    return generate_response(sys_p, usr_p, model=model_name, max_new_tokens=max_new_tokens, pipeline=pipeline, log_filepath=log_filepath)