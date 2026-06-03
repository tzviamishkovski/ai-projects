import os
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
client = OpenAI(api_key=api_key)

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'commandPrompt.md')

def load_documentation(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            print(f"Documentation loaded successfully from {path}. Length: {len(content)} characters")
            return content
    except Exception as e:
        print(f"CRITICAL ERROR loading documentation from {path}: {e}")
        return ""


# Load documentation contents
PROMPT_PATH_CONTEXT = load_documentation(PROMPT_PATH)
if not PROMPT_PATH_CONTEXT:
    print("WARNING: No prompt documentation context loaded!")

MODEL = "gpt-4o"
 
 
def respond(message: str, history: list) -> str:
    """
    Called by Gradio on each user message.
    `history` is a list of {"role": ..., "content": ...} dicts (Gradio messages format).
    """
    # System prompt first, then conversation history, then new user message
    messages = [{"role": "system", "content": PROMPT_PATH_CONTEXT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})
 
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
 
    return response.choices[0].message.content
 
 
def main():
    demo = gr.ChatInterface(
        fn=respond,
        title="GPT-4o Chat",
        description=f"Powered by {MODEL}"
    )
    demo.launch()
 
 
if __name__ == "__main__":
    main()
 