from pydantic import BaseModel, Field
from typing import Optional
from openai import OpenAI


class BaseLLM:
    def __init__(self, client, model_name: Optional[str] = None):
        self.client = client
        self.model_name = model_name
    
    def parse(self, prompt: str, schema: BaseModel, temperature: Optional[float] = None):
        ...

class OpenAILLM(BaseLLM):
    def __init__(self, client: OpenAI, model_name: Optional[str] = 'gpt-4o'):
        self.client = client
        self.model_name = model_name
            
    def parse(self, prompt: str, schema, temperature: Optional[float] = None):
        kwargs = dict(model=self.model_name, input=[{"role": "user", "content": prompt}], text_format=schema)
        if temperature is not None:
            kwargs["temperature"] = temperature
        return self.client.responses.parse(**kwargs).output_parsed

