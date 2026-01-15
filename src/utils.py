import os
from openai import OpenAI
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from .prompts import create_message

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_HF_pipeline(model_name: str):
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
        max_new_tokens=256,   # Enough for JSON output
        temperature=0.1,      # Keep it factual/consistent
        do_sample=True,
        return_full_text=False
    )
    return pipe


def generate_response(system_prompt_str: str, user_prompt_str: str, model, temperature: float = 0.3,
    top_p: float = 0.95):

    message = create_message(system_prompt_str, user_prompt_str)
    if model.startswith("gpt"):
        resp = client.chat.completions.create(
            model=model,
            messages=message,
            temperature=temperature,
            top_p=top_p,
        ).choices[0].message.content
    else:
        pipe = get_HF_pipeline(model)
        prompt = pipe.tokenizer.apply_chat_template(
        message, 
        tokenize=False, 
        add_generation_prompt=True
        )
        outputs = pipe(prompt)
        resp = outputs[0]['generated_text'] 
        
    return resp


    