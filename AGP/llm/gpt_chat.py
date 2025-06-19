from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from openai import AsyncOpenAI
from httpx import Timeout

from AGP.llm.format import Message
from AGP.llm.price import cost_count
from AGP.llm.llm import LLM
from AGP.llm.llm_registry import LLMRegistry

MINE_BASE_URL = ''
MINE_API_KEYS = ''


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(5))
async def achat(
    model: str = "gpt-4o-mini",
    msg: List[Dict] = [],):
    request_url = MINE_BASE_URL
    authorization_key = MINE_API_KEYS
    try:
        aclient = AsyncOpenAI(
            base_url=request_url, 
            api_key=authorization_key,  
            timeout=Timeout(100.0)
        )
        completion = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=msg,
            max_tokens=10000,
        )
        prompt = "".join([item['content'] for item in msg])
        cost_count(prompt, completion.choices[0].message.content, "gpt-4o-mini")
        return completion.choices[0].message.content  
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass