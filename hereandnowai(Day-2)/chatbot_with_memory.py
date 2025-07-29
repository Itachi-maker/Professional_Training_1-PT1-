from openai import OpenAI
from dotenv import load_dotenv
import os
from prompts import ai_motivational_speaker
load_dotenv()
api_key = os.getenv("GEMENI_API_KEY")
model = "gemini-2.5-flash-lite"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
client = OpenAI(base_url=base_url, api_key=api_key)
ai_motivational_speaker= ai_motivational_speaker
def get_response(message, history):
    messages = [{"role": "system", "content": ai_motivational_speaker}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(model=model, messages=messages)
    ai_response = response.choices[0].message.content
    return ai_response
if __name__ == "__main__":
    print(get_response("Hello Caramel AI, what does dicipline stands for?", []))