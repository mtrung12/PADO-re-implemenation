from .prompts import (
    HIGH_INDUCE,
    LOW_INDUCE,
    ZERO_INFERENCE_USER_PROMPT,
    ZERO_INFERENCE_SYSTEM_PROMPT,
    ONE_INFERENCE_USER_PROMPT,
    ONE_INFERENCE_SYSTEM_PROMPT,
    COT_INFERENCE_USER_PROMPT,
    COT_INFERENCE_SYSTEM_PROMPT,
    PADO_INFERENCE_USER_PROMPT,
    PADO_INFERENCE_SYSTEM_PROMPT
)
from .utils import generate_response



def generate_explaination(ctrait, ctext, induce='high', prompt_type='pado'):    
    if prompt_type == 'pado': 
        usr_p = PADO_INFERENCE_USER_PROMPT.format(trait = ctrait, text = ctext)
        if induce == 'high':  
            sys_p = PADO_INFERENCE_SYSTEM_PROMPT.format(induce_level = HIGH_INDUCE[ctrait])
        else:
            sys_p = PADO_INFERENCE_SYSTEM_PROMPT.format(induce_level = LOW_INDUCE[ctrait])
    elif prompt_type == 'zero':
        sys_p = ZERO_INFERENCE_SYSTEM_PROMPT.format(trait = ctrait)
        usr_p = ZERO_INFERENCE_USER_PROMPT.format(text = ctext)
    elif prompt_type == 'one':
        sys_p = ONE_INFERENCE_SYSTEM_PROMPT.format(
            trait = ctrait,
            # currently experiment with a single possible false example (TODO)
            example_text = "I love spending time with my friends and meeting new people.",
            example_label = "high"
        )
        usr_p = ONE_INFERENCE_USER_PROMPT.format(text = ctext)
    elif prompt_type == 'cot':
        sys_p = COT_INFERENCE_SYSTEM_PROMPT.format(trait = ctrait)
        usr_p = COT_INFERENCE_USER_PROMPT.format(text = ctext)
    else:
        raise ValueError("Invalid prompt_type. Choose from 'pado', 'zero', 'one', or 'cot'.")

    return generate_response(sys_prompt_str=sys_p, user_prompt_str=usr_p)