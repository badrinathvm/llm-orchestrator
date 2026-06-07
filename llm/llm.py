import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        load_dotenv()
        self.model = model
        self.temperature = temperature

    def get_llm(self):
        try:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            llm = ChatOpenAI(api_key=self.openai_api_key, model=self.model, temperature=self.temperature)
            return llm
        except Exception as e:
            raise ValueError(f"Error occurred with exception: {e}")
