from typing import Optional
from class_registry import ClassRegistry

from AGP.llm.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None or model_name=="":
            model_name = "gpt-4o"

        if model_name == 'mock':
            model = cls.registry.get(model_name)
        else: # any version of GPTChat like "gpt-4o"
            model = cls.registry.get('GPTChat', model_name)

        return model
