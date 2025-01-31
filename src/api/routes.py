from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, FastAPI, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Optional, List, Union, AsyncGenerator
import asyncio
import json
import time
import logging
from pydantic import BaseModel, Field
import torch
from ..model import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_model_manager(request: Request):
    """Dependency to get model manager from app state"""
    if not hasattr(request.app.state, "model_manager") or not request.app.state.model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded or initialized")
    return request.app.state.model_manager


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


async def format_stream_response(generator: AsyncGenerator[str, None]):
    """Format streaming response in OpenAI-compatible format"""
    try:
        async for chunk in generator:
            if chunk:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        await asyncio.sleep(0)  # Yield control


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


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, background_tasks: BackgroundTasks, model_manager: ModelManager = Depends(get_model_manager)):
    try:
        # Convert chat format to prompt
        prompt = format_chat_with_context(request.messages)
        logger.info(f"Chat request received. Prompt: {prompt[:100]}...")

        # Safely handle temperature and sampling
        temperature = request.temperature if request.temperature is not None else 0.7
        do_sample = temperature > 0

        generator = model_manager.inference.generate_stream(
            prompt=prompt,
            generation_config={"temperature": temperature, "max_new_tokens": request.max_tokens or 4096, "do_sample": do_sample},
        )

        if request.stream:
            return StreamingResponse(format_stream_response(generator), media_type="text/event-stream")
        else:
            response_text = ""
            start_time = time.time()
            async for chunk in generator:
                response_text += chunk
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Chat completion completed in {elapsed_time:.2f} seconds.")

            return {
                "id": f"chatcmpl-{time.time_ns()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "deepseek-coder",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(request.messages), "completion_tokens": len(response_text.split()), "total_tokens": len(request.messages) + len(response_text.split())},
            }
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/completions")
async def completions(request: CompletionRequest, background_tasks: BackgroundTasks, model_manager: ModelManager = Depends(get_model_manager)):
    try:
        if not model_manager.inference:
            raise HTTPException(status_code=503, detail="Model inference not initialized")
        logger.info(f"Completion request received. Prompt: {request.prompt[:100]}...")

        # Safely handle temperature and sampling
        temperature = request.temperature if request.temperature is not None else 0.7
        do_sample = temperature > 0

        generator = model_manager.inference.generate_stream(prompt=request.prompt, generation_config={"temperature": temperature, "max_new_tokens": request.max_tokens or 4096, "do_sample": do_sample})

        if request.stream:
            return StreamingResponse(generate_stream_response(generator), media_type="text/event-stream")
        else:
            response_text = ""
            start_time = time.time()
            async for chunk in generator:
                response_text += chunk
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Completion completed in {elapsed_time:.2f} seconds.")

            return {"id": f"cmpl-{time.time_ns()}", "object": "text_completion", "created": int(time.time()), "choices": [{"text": response_text, "index": 0, "finish_reason": "stop"}]}

    except Exception as e:
        logger.error(f"Completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/models")
async def list_models():
    return {"data": [{"id": "deepseek-coder", "object": "model", "created": int(time.time()), "owned_by": "user"}]}


@router.get("/health")
async def health_check(model_manager: ModelManager = Depends(get_model_manager)):
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    memory_info = torch.cuda.mem_get_info()
    memory_allocated = torch.cuda.memory_allocated()

    return {"status": "healthy", "model_loaded": True, "gpu_memory": {"free": memory_info[0] / (1024**3), "total": memory_info[1] / (1024**3), "used": memory_allocated / (1024**3)}}


def setup_routes(app: FastAPI):
    """Setup all routes for the application"""
    app.include_router(router)


