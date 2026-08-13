from pathlib import Path
import sys
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))


from openai import AsyncOpenAI,APITimeoutError,APIConnectionError,APIError
from config import settings
import structlog
import asyncio


logger = structlog.get_logger(__name__)

class LLMClient():
     
    def __init__(self):
         self._client = AsyncOpenAI(
             api_key = settings.GLM_API_KEY,
             base_url = settings.GLM_BASE_URL,
             timeout = settings.LLM_TIMEOUT,
             max_retries = 0 
         )
         
    async def chat(
        self,
        messages:list,
        tools:list | None = None,
        model:str | None = None,
        temperature:float | None = None,
        max_tokens:int | None = None,
        stream:bool = False,
    ):
        
        model = model or settings.GLM_MODEL_NAME
        temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        kwargs = {
            "messages" : messages,
            "model" : model,
            "temperature" : temperature,
            "max_tokens" : max_tokens,
            "stream" : False
        }
        
        if tools:
            kwargs["tools"] = tools
        
        for attempt in range(settings.TOOL_MAX_RETRIES):
            try:
                resp = await self._client.chat.completions.create(**kwargs)
                # 成功，打印日志并返回
                logger.debug(
                    "llm_chat_success",
                    model = model,
                    attempt = attempt+1,
                    tokens = resp.usage.total_tokens if resp.usage else None,  
                )
                return resp
            except (APIConnectionError,APITimeoutError) as e:
                # 网络类型错误，可重试
                wait_time = attempt + 1
                logger.warning(
                    "llm_retry_error",
                    error=str(e),
                    wait_time=wait_time,
                )
                if attempt + 1 < settings.LLM_MAX_RETRIES:
                    await asyncio.sleep(wait_time)
                
            except APIError as e:
                # 认证错误，不重试
                logger.error(
                    "llm_api_error",
                    error=str(e)
                    )
                raise

        
        
    
    async def stream_chat(
            self,
            messages:list,
            tools:list | None = None,
            model:str | None = None,
            temperature:float | None = None,
            max_tokens:int | None = None,
            stream:bool = True,
        ):
            
            model = model or settings.GLM_MODEL_NAME
            temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
            max_tokens = max_tokens or settings.LLM_MAX_TOKENS
            
            kwargs = {
                "messages" : messages,
                "model" : model,
                "temperature" : temperature,
                "max_tokens" : max_tokens,
                "stream" : True
            }
            
            if tools:
                kwargs["tools"] = tools
            
            for attempt in range(settings.LLM_MAX_RETRIES):
                try:
                    stream = self._client.chat.completions.create(**kwargs)
                    async for chunk in stream:
                            yield chunk
                    logger.debug(
                        "llm_stream_chat_success",
                        attempt = attempt + 1,
                        
                    )
                    return
                except (APITimeoutError,APIConnectionError) as e:
                    wait_time = attempt + 1
                    
                    logger.warning(
                        "llm_retry_error",
                        error=str(e),
                        wait_time = wait_time
                    )
                    
                    if attempt + 1 < settings.LLM_MAX_RETRIES:
                        await asyncio.sleep(wait_time)
                        
                except APIError as e:
                    logger.error(
                        "llm_api_error",
                        error = str(e)
                    )
                    raise
            
        
        
    async def lite_chat(
        self,
        sys_prompt:str,
        user_content:str,
        temperature:float = 0,
        max_tokens:int = 1024,
        stream:bool = True,
    ):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        
        resp = self.chat(
            messages = messages,
            temperature = temperature,
            max_tokens = max_tokens,
            model = settings.GLM_LITE_MODEL
        )
        
        return resp.choices[0].message.content
    
    async def close(self):
        await self._client.close()
        
_llm_client:LLMClient | None = None        

def get_llm_client():
    
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client