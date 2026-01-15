import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(
    system_prompt_str: str,
    user_prompt_str: str,
    model: str = "gpt-4o",
    temperature: float = 0.3,
    top_p: float = 0.95,
) -> str:
    """
    Takes system_prompt_str and user_prompt_str,
    performs a chat completion, and returns the resulting text.
    """
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text", 
                    "text": system_prompt_str
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": user_prompt_str
                }
            ],
        },
    ]
    
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
    ).choices[0].message.content

    return resp

def extract_label_from_response(response: str) -> str:
    """
    Extracts the label ('high' or 'low') from the model
    response string.
    """
    