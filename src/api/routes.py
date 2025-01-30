from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, Optional, List, Union
import asyncio
import json
import time
from pydantic import BaseModel, Field
import torch
from ..model import ModelManager

router = APIRouter()
model_manager = ModelManager()

class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(4096, ge=1, le=131072)
    stream: Optional[bool] = True

class CompletionRequest(BaseModel):
    prompt: Union[str, List[str]]
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(4096, ge=1, le=131072)
    stream: Optional[bool] = True

async def generate_stream_response(generator):
    try:
        async for chunk in generator:
            if chunk:
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        await asyncio.sleep(0)  # Yield control

# src/api/routes.py

@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    try:
        # Convert chat format to prompt with proper handling of file context
        prompt = format_chat_with_context(request.messages)
        
        generator = model_manager.inference.generate_stream(
            prompt=prompt,
            generation_config={
                "temperature": request.temperature,
                "max_new_tokens": request.max_tokens,
                "do_sample": True if request.temperature > 0 else False
            }
        )

        if request.stream:
            return StreamingResponse(
                format_stream_response(generator),
                media_type="text/event-stream"
            )
        else:
            # Collect full response
            response_text = ""
            async for chunk in generator:
                response_text += chunk
            
            return {
                "id": f"chatcmpl-{time.time_ns()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "deepseek-coder",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(request.messages),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(request.messages) + len(response_text.split())
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def format_stream_response(generator):
    """Format streaming response in OpenAI-compatible format"""
    try:
        async for chunk in generator:
            if chunk:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

def format_chat_with_context(messages: List[ChatMessage]) -> str:
    """Format chat messages with file context"""
    formatted = []
    for msg in messages:
        if msg.role == "system":
            formatted.append(f"System: {msg.content}")
        elif msg.role == "user":
            formatted.append(f"Human: {msg.content}")
        elif msg.role == "assistant":
            formatted.append(f"Assistant: {msg.content}")
    
    return "\n".join(formatted)

@router.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    background_tasks: BackgroundTasks
):
    try:
        generator = model_manager.inference.generate_stream(
            prompt=request.prompt,
            generation_config={
                "temperature": request.temperature,
                "max_new_tokens": request.max_tokens,
                "do_sample": True if request.temperature > 0 else False
            }
        )

        if request.stream:
            return StreamingResponse(
                generate_stream_response(generator),
                media_type="text/event-stream"
            )
        else:
            response_text = ""
            async for chunk in generator:
                response_text += chunk
            
            return {
                "id": f"cmpl-{time.time_ns()}",
                "object": "text_completion",
                "created": int(time.time()),
                "choices": [{
                    "text": response_text,
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/v1/models")
async def list_models():
    return {
        "data": [{
            "id": "deepseek-coder",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "user"
        }]
    }

@router.get("/health")
async def health_check():
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    memory_info = torch.cuda.mem_get_info()
    memory_allocated = torch.cuda.memory_allocated()
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "gpu_memory": {
            "free": memory_info[0] / (1024**3),
            "total": memory_info[1] / (1024**3),
            "used": memory_allocated / (1024**3)
        }
    }

def setup_routes(app: FastAPI):
    """Setup all routes for the application"""
    app.include_router(router)
