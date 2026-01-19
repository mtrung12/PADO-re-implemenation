import os
from openai import OpenAI
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from .prompts import create_message_openai, create_message_HF
from typing import List, Union
from tqdm import tqdm


def get_HF_pipeline(model_name: str, max_new_tokens: int = 512, batch_size: int = 1):
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
        return_full_text=False,
        batch_size=batch_size
    )
    return pipe


def generate_response(system_prompt_str: str, user_prompt_str: Union[str, List[str]], model: str, temperature: float = 0.3,
    top_p: float = 0.95, max_new_tokens: int = None, pipeline = None) -> Union[str, List[str]]:

    is_batch = isinstance(user_prompt_str, list)

    if model.startswith("gpt"):
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        params = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_new_tokens is not None:
            params["max_tokens"] = max_new_tokens

        if is_batch:
            responses = []
            for u_prompt in tqdm(user_prompt_str, desc="Processing GPT requests"):
                message = create_message_openai(system_prompt_str, u_prompt)
                params["messages"] = message
                resp = client.chat.completions.create(**params).choices[0].message.content
                responses.append(resp)
            return responses
        else:
            message = create_message_openai(system_prompt_str, user_prompt_str)
            params["messages"] = message
            return client.chat.completions.create(**params).choices[0].message.content
    else:
        if is_batch:
            prompts = []
            for u_prompt in user_prompt_str:
                message = create_message_HF(system_prompt_str, u_prompt)
                prompts.append(pipeline.tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                ))
            
            outputs = pipeline(prompts)
            return [out[0]['generated_text'] for out in outputs]
        else:
            message = create_message_HF(system_prompt_str, user_prompt_str)
            prompt = pipeline.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            outputs = pipeline(prompt)
            return outputs[0]['generated_text']


    