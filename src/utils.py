import os
from openai import OpenAI
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from .prompts import create_message_openai, create_message_HF
from typing import List, Union
from tqdm import tqdm
import google.generativeai as genai
import time
import json

def get_HF_pipeline(model_name: str, max_new_tokens: int = 512):
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
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True,
        return_full_text=False,
    )
    return pipe

def extract_json_from_response(response_text: str) -> str:
    """
    Extracts a JSON object from a string that might be wrapped in Markdown.
    """
    if not isinstance(response_text, str):
        return ""
        
    # Find the first '{' and the last '}'
    start_index = response_text.find('{')
    end_index = response_text.rfind('}')

    if start_index != -1 and end_index != -1 and start_index < end_index:
        # Return the substring that is the JSON object
        return response_text[start_index:end_index+1]
    else:
        # Return an empty string if no valid JSON object is found
        return ""

def _log_to_file(log_filepath: str, system_prompt: str, user_prompt: Union[str, List[str]], response: Union[str, List[str]]):
    if not log_filepath:
        return
    try:
        with open(log_filepath, 'a', encoding='utf-8') as f:
            f.write(f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n")
            
            if isinstance(user_prompt, list):
                f.write(f"--- USER PROMPT (BATCH) ---\n")
                for i, prompt_item in enumerate(user_prompt):
                    f.write(f"  --- Item {i+1}/{len(user_prompt)} ---\n")
                    f.write(f"{prompt_item}\n")
            else:
                f.write(f"--- USER PROMPT ---\n{user_prompt}\n")
            f.write("\n")
            
            response_str = str(response)
            f.write(f"--- LLM RESPONSE ---\n{response_str}\n")
            f.write("="*80 + "\n\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")

def generate_response(system_prompt_str: str, user_prompt_str: Union[str, List[str]], model: str, temperature: float = 0.3,
    top_p: float = 0.95, max_new_tokens: int = None, pipeline = None, log_filepath: str = None) -> Union[str, List[str]]:

    is_batch_of_individual_prompts = isinstance(user_prompt_str, list)
    is_manual_batch_model = model.startswith("gpt") or model.startswith("gemini")

    if is_manual_batch_model:
        # Logic for GPT and Gemini (sequential with sleep)
        load_dotenv()
        if model.startswith("gpt"):
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            params = {
                "model": model, "temperature": temperature, "top_p": top_p,
            }
            if max_new_tokens is not None:
                params["max_tokens"] = max_new_tokens
            
            message = create_message_openai(system_prompt_str, user_prompt_str)
            params["messages"] = message
            try:
                content = client.chat.completions.create(**params).choices[0].message.content
                _log_to_file(log_filepath, system_prompt_str, user_prompt_str, content)
                return content
            except Exception as e:
                error_msg = f"Error: {e}"
                _log_to_file(log_filepath, system_prompt_str, user_prompt_str, error_msg)
                return error_msg
            finally:
                time.sleep(50)

        elif model.startswith("gemini"):
            api_key = os.getenv("GEMINI_KEY")
            if not api_key:
                raise ValueError("GEMINI_KEY not found in .env file.")
            genai.configure(api_key=api_key)
            
            generation_config = genai.GenerationConfig(temperature=temperature, top_p=top_p, max_output_tokens=max_new_tokens)
            gemini_model = genai.GenerativeModel(model, system_instruction=system_prompt_str, generation_config=generation_config)
            
            try:
                response_obj = gemini_model.generate_content(user_prompt_str)
                content = response_obj.text
                _log_to_file(log_filepath, system_prompt_str, user_prompt_str, content)
                return content
            except Exception as e:
                error_msg = f"Error: {e}"
                _log_to_file(log_filepath, system_prompt_str, user_prompt_str, error_msg)
                return error_msg
            finally:
                time.sleep(50)
    
    else: # HuggingFace models
        if is_batch_of_individual_prompts:
            prompts = [
                pipeline.tokenizer.apply_chat_template(
                    create_message_HF(system_prompt_str, u_prompt),
                    tokenize=False, add_generation_prompt=True
                ) for u_prompt in user_prompt_str
            ]
            outputs = pipeline(prompts)
            results = [out[0]['generated_text'] for out in outputs]
            _log_to_file(log_filepath, system_prompt_str, user_prompt_str, results)
            return results
        else: # Single HF prompt
            message = create_message_HF(system_prompt_str, user_prompt_str)
            prompt = pipeline.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            outputs = pipeline(prompt)
            result = outputs[0]['generated_text']
            _log_to_file(log_filepath, system_prompt_str, user_prompt_str, result)
            return result