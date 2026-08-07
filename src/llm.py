import os

from dotenv import load_dotenv

from llm_provider import OpenAICompatibleProvider

load_dotenv()

llm = OpenAICompatibleProvider(
    model="Qwen3.6-35B-A3B-4bit",
    base_url="http://192.168.178.45:11434/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)
