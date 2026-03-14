api_key = "sk-108cf15a7a0b4d12931f59caf1963f5f"

def load_dotenv():
    """Load environment variables from a .env file."""
    return api_key
    
from openai import OpenAI

def generate_with_ds(prompt):
    
    client = OpenAI(api_key=load_dotenv(), base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature= 0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    return response.choices[0].message.content