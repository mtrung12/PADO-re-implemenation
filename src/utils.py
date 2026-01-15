import os
from openai import OpenAI
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from .prompts import create_message



def get_HF_pipeline(model_name: str, max_new_tokens: int = 256):
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    # Create generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,   # Use the passed argument
        temperature=0.1,      # Keep it factual/consistent
        do_sample=True,
        return_full_text=False
    )
    return pipe


def generate_response(system_prompt_str: str, user_prompt_str: str, model, temperature: float = 0.3,
    top_p: float = 0.95, max_new_tokens: int = None):

    message = create_message(system_prompt_str, user_prompt_str)
    if model.startswith("gpt"):
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        params = {
            "model": model,
            "messages": message,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_new_tokens is not None:
            params["max_tokens"] = max_new_tokens

        resp = client.chat.completions.create(**params).choices[0].message.content
    else:
        pipe = get_HF_pipeline(model, max_new_tokens=max_new_tokens if max_new_tokens is not None else 256)
        prompt = pipe.tokenizer.apply_chat_template(
        message, 
        tokenize=False, 
        add_generation_prompt=True
        )
        outputs = pipe(prompt)
        resp = outputs[0]['generated_text'] 
        
    return resp


    