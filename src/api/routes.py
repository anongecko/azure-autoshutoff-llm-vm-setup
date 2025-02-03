from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks, FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Optional, List, Union, Any, AsyncGenerator
import asyncio
import json
import time
import logging
from pydantic import BaseModel, Field, validator, model_validator
import torch
from pathlib import Path
from ..model import ModelManager
from ..utils.memory_manager import OptimizedMemoryManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Enhanced configuration loading with caching
_config_cache = {}


def get_config() -> tuple[Dict, Dict]:
    """Get cached configurations"""
    if not _config_cache:
        try:
            config_dir = Path("config")
            with open(config_dir / "api_config.json") as f:
                _config_cache["api"] = json.load(f)
            with open(config_dir / "model_config.json") as f:
                _config_cache["model"] = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise RuntimeError("Failed to load configuration files")
    return _config_cache["api"], _config_cache["model"]


API_CONFIG, MODEL_CONFIG = get_config()


class ChatMessage(BaseModel):
    """Enhanced chat message model"""

    role: str = Field(..., description="Message sender role")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name for the message sender")

    @validator("role")
    def validate_role(cls, v):
        allowed_roles = {"system", "user", "assistant"}
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v


class ChatRequest(BaseModel):
    """Enhanced chat completion request"""
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(MODEL_CONFIG["generation"]["temperature"], ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(MODEL_CONFIG["attention"]["max_new_tokens"], ge=1)
    stream: Optional[bool] = Field(MODEL_CONFIG["code_specific"]["response_format"]["streaming_chunks"])
    top_p: Optional[float] = Field(MODEL_CONFIG["generation"]["top_p"], ge=0.0, le=1.0)
    presence_penalty: Optional[float] = Field(MODEL_CONFIG["generation"]["presence_penalty"], ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(MODEL_CONFIG["generation"]["frequency_penalty"], ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    language: Optional[str] = Field(None, description="Programming language for code-specific prompting")

    @model_validator(mode='after')
    def validate_request(self) -> 'ChatRequest':
        # Validate total message length
        if hasattr(self, "messages"):
            total_chars = sum(len(msg.content) for msg in self.messages)
            approx_tokens = total_chars // 4  # Rough estimate
            if approx_tokens > MODEL_CONFIG["max_sequence_length"]:
                raise ValueError(f"Total message length exceeds model's context window")

        # Adjust generation parameters for code
        if self.language:
            self.temperature = min(self.temperature or 0.7, 0.8)
            self.top_p = min(self.top_p or 0.95, 0.95)

        return self

class CompletionRequest(BaseModel):
    """Enhanced text completion request"""
    prompt: Union[str, List[str]]
    temperature: Optional[float] = Field(MODEL_CONFIG["generation"]["temperature"], ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(MODEL_CONFIG["attention"]["max_new_tokens"], ge=1)
    stream: Optional[bool] = Field(True)
    top_p: Optional[float] = Field(MODEL_CONFIG["generation"]["top_p"], ge=0.0, le=1.0)
    presence_penalty: Optional[float] = Field(MODEL_CONFIG["generation"]["presence_penalty"], ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(MODEL_CONFIG["generation"]["frequency_penalty"], ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    language: Optional[str] = Field(None, description="Programming language for code-specific prompting")

    @model_validator(mode='after')
    def validate_request(self) -> 'CompletionRequest':
        # Validate prompt length
        if isinstance(self.prompt, list):
            total_chars = sum(len(p) for p in self.prompt)
        else:
            total_chars = len(self.prompt)
        approx_tokens = total_chars // 4
        if approx_tokens > MODEL_CONFIG["max_sequence_length"]:
            raise ValueError(f"Prompt length exceeds model's context window")

        # Adjust for code completion
        if self.language:
            self.temperature = min(self.temperature or 0.7, 0.8)
            self.top_p = min(self.top_p or 0.95, 0.95)

        return self

async def get_model_manager(request: Request) -> ModelManager:
    """Get and validate model manager"""
    if not hasattr(request.app.state, "model_manager"):
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    model_manager = request.app.state.model_manager
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_manager


async def get_memory_manager(request: Request) -> OptimizedMemoryManager:
    """Get memory manager"""
    if not hasattr(request.app.state, "memory_manager"):
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    return request.app.state.memory_manager


def detect_language(content: str) -> Optional[str]:
    """Detect programming language from content"""
    # Common language indicators
    indicators = {
        "python": ["def ", "class ", "import ", "from ", "@", "async def"],
        "javascript": ["function", "const ", "let ", "var ", "=>", "async "],
        "typescript": ["interface ", "type ", "enum ", ": string", ": number"],
        "html": ["<!DOCTYPE", "<html", "<div", "<script", "<style"],
        "css": ["{", ":", "@media", "@keyframes"],
        "sql": ["SELECT ", "INSERT ", "UPDATE ", "DELETE ", "CREATE TABLE"],
    }

    content_lower = content.lower()
    for lang, patterns in indicators.items():
        if any(pattern.lower() in content_lower for pattern in patterns):
            return lang
    return None


def format_chat_prompt(messages: List[ChatMessage], language: Optional[str] = None) -> str:
    """Format chat messages with enhanced code handling"""
    formatted = []

    # Add language context if provided
    if language:
        formatted.append(f"System: You are a coding assistant specializing in {language}. Please provide clear, well-structured code with proper indentation.")

    for msg in messages:
        prefix = {"system": "System: ", "user": "Human: ", "assistant": "Assistant: "}.get(msg.role, "")

        content = msg.content
        if MODEL_CONFIG["code_specific"]["preserve_indentation"]:
            # Preserve code structure
            if detect_language(content):
                content = "\n".join(line for line in content.splitlines())

        formatted.append(f"{prefix}{content}")

    return "\n".join(formatted)


async def format_code_chunk(chunk: str, language: Optional[str] = None, include_line_numbers: bool = False) -> str:
    """Enhanced code chunk formatting"""
    if not chunk.strip():
        return chunk

    is_code = detect_language(chunk) is not None
    if is_code:
        # Apply language-specific formatting
        if language:
            chunk = await apply_language_formatting(chunk, language)

        if include_line_numbers and MODEL_CONFIG["code_specific"]["enable_line_numbers"]:
            lines = chunk.splitlines()
            chunk = "\n".join(f"{i + 1:4d} │ {line}" for i, line in enumerate(lines))

        # Preserve indentation
        if MODEL_CONFIG["code_specific"]["preserve_indentation"]:
            chunk = "\n".join(line for line in chunk.splitlines())

    return chunk


async def apply_language_formatting(code: str, language: str) -> str:
    """Apply language-specific code formatting"""
    if language == "python":
        # Ensure consistent Python indentation
        lines = code.splitlines()
        formatted = []
        indent_level = 0

        for line in lines:
            stripped = line.strip()
            # Adjust indent level based on content
            if stripped.endswith(":"):
                formatted.append("    " * indent_level + stripped)
                indent_level += 1
            elif stripped in ["return", "break", "continue", "pass"]:
                indent_level = max(0, indent_level - 1)
                formatted.append("    " * indent_level + stripped)
            else:
                formatted.append("    " * indent_level + stripped)

        return "\n".join(formatted)

    elif language in ["javascript", "typescript"]:
        # Basic JS/TS formatting
        return code.replace(");", ");\n").replace("{ ", "{\n    ").replace(" }", "\n}")

    return code


async def format_streaming_response(generator: AsyncGenerator[str, None], request_id: str, language: Optional[str] = None) -> AsyncGenerator[str, None]:
    """Enhanced streaming response formatting"""
    try:
        buffer = ""
        async for chunk in generator:
            if chunk:
                # Buffer the output for better code block detection
                buffer += chunk
                if len(buffer) > 100 or buffer.endswith(("\n", "}", ";")):
                    # Format buffered content
                    formatted_chunk = await format_code_chunk(buffer, language=language, include_line_numbers=MODEL_CONFIG["code_specific"]["enable_line_numbers"])

                    yield "data: " + json.dumps({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": MODEL_CONFIG["model_name"],
                        "choices": [{"index": 0, "delta": {"content": formatted_chunk}, "finish_reason": None}],
                    }) + "\n\n"
                    buffer = ""

        # Flush remaining buffer
        if buffer:
            formatted_chunk = await format_code_chunk(buffer, language=language, include_line_numbers=MODEL_CONFIG["code_specific"]["enable_line_numbers"])
            yield "data: " + json.dumps({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": MODEL_CONFIG["model_name"],
                "choices": [{"index": 0, "delta": {"content": formatted_chunk}, "finish_reason": None}],
            }) + "\n\n"

        # Send completion
        yield "data: " + json.dumps({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_CONFIG["model_name"],
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }) + "\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield "data: " + json.dumps({"error": str(e)}) + "\n\n"


def prepare_generation_config(request: Union[ChatRequest, CompletionRequest], language: Optional[str] = None) -> Dict[str, Any]:
    """Enhanced generation configuration preparation"""
    config = {
        "temperature": request.temperature,
        "max_new_tokens": request.max_tokens,
        "do_sample": request.temperature > 0,
        "top_p": request.top_p,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "stop_sequences": request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
        "repetition_penalty": MODEL_CONFIG["generation"]["repetition_penalty"],
        "typical_p": MODEL_CONFIG["generation"]["typical_p"],
        "top_k": MODEL_CONFIG["generation"]["top_k"],
    }

    if MODEL_CONFIG["code_specific"]["enable_code_completion"]:
        config.update({"preserve_indentation": True, "language_specific_prompting": True, "language": language})

        # Language-specific adjustments
        if language == "python":
            config["repetition_penalty"] = 1.2  # Stronger for Python
        elif language in ["javascript", "typescript"]:
            config["repetition_penalty"] = 1.15  # Moderate for JS/TS

    return config


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest, background_tasks: BackgroundTasks, model_manager: ModelManager = Depends(get_model_manager), memory_manager: OptimizedMemoryManager = Depends(get_memory_manager)
):
    """Enhanced chat completions handler"""
    try:
        # Detect language if not specified
        language = request.language
        if not language:
            last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
            if last_user_message:
                language = detect_language(last_user_message)

        # Format prompt with language context
        prompt = format_chat_prompt(request.messages, language)

        # Prepare generation config with language optimization
        generation_config = prepare_generation_config(request, language)
        request_id = f"chatcmpl-{int(time.time() * 1000)}"

        # Verify memory state
        if not await memory_manager.verify_memory_state():
            raise HTTPException(status_code=503, detail="Insufficient memory available", headers={"Retry-After": "30"})

        # Get generator
        generator = model_manager.inference.generate_stream(prompt=prompt, generation_config=generation_config)

        # Handle streaming
        if request.stream:
            return StreamingResponse(format_streaming_response(generator, request_id, language), media_type="text/event-stream")

        # Collect full response
        response_text = ""
        start_time = time.time()

        async for chunk in generator:
            formatted_chunk = await format_code_chunk(chunk, language)
            response_text += formatted_chunk

        generation_time = time.time() - start_time
        logger.info(f"Completion generated in {generation_time:.2f}s")

        # Schedule cleanup
        background_tasks.add_task(memory_manager._aggressive_cleanup)

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_CONFIG["model_name"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": len(prompt) // 4,  # Approximate
                "completion_tokens": len(response_text) // 4,
                "total_tokens": (len(prompt) + len(response_text)) // 4,
            },
            "language": language,
        }

    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/completions")
async def completions(
    request: CompletionRequest, background_tasks: BackgroundTasks, model_manager: ModelManager = Depends(get_model_manager), memory_manager: OptimizedMemoryManager = Depends(get_memory_manager)
):
    """Enhanced text completions handler"""
    try:
        # Handle prompt and detect language
        if isinstance(request.prompt, list):
            prompt = "\n".join(request.prompt)
        else:
            prompt = request.prompt

        language = request.language or detect_language(prompt)
        logger.info(f"Completion request received - Length: {len(prompt)}, Language: {language}")

        # Prepare enhanced generation config
        generation_config = prepare_generation_config(request, language)
        request_id = f"cmpl-{int(time.time() * 1000)}"

        # Verify memory state
        if not await memory_manager.verify_memory_state():
            raise HTTPException(status_code=503, detail="Insufficient memory available", headers={"Retry-After": "30"})

        # Get generator
        generator = model_manager.inference.generate_stream(prompt=prompt, generation_config=generation_config)

        # Handle streaming
        if request.stream:
            return StreamingResponse(format_streaming_response(generator, request_id, language), media_type="text/event-stream")

        # Collect full response
        response_text = ""
        start_time = time.time()

        async for chunk in generator:
            formatted_chunk = await format_code_chunk(chunk, language)
            response_text += formatted_chunk

        generation_time = time.time() - start_time
        logger.info(f"Completion generated in {generation_time:.2f}s")

        # Schedule cleanup
        background_tasks.add_task(memory_manager._aggressive_cleanup)

        return {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": MODEL_CONFIG["model_name"],
            "choices": [{"text": response_text, "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(response_text) // 4, "total_tokens": (len(prompt) + len(response_text)) // 4},
            "language": language,
        }

    except Exception as e:
        logger.error(f"Completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/models")
async def list_models():
    """List available models with enhanced information"""
    return {
        "data": [
            {
                "id": MODEL_CONFIG["model_name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user",
                "permission": [],
                "root": None,
                "parent": None,
                "context_window": MODEL_CONFIG["max_sequence_length"],
                "capabilities": {
                    "code_completion": MODEL_CONFIG["code_specific"]["enable_code_completion"],
                    "syntax_highlighting": MODEL_CONFIG["code_specific"]["syntax_highlighting"],
                    "streaming": True,
                    "languages": ["python", "javascript", "typescript", "html", "css", "sql"],
                },
            }
        ]
    }


@router.get("/health")
async def health_check(model_manager: ModelManager = Depends(get_model_manager), memory_manager: OptimizedMemoryManager = Depends(get_memory_manager)):
    """Enhanced health check endpoint"""
    try:
        # Get comprehensive status
        model_info = model_manager.get_model_info()
        memory_info = memory_manager.get_memory_info()
        hardware_info = {"device_map": MODEL_CONFIG["hardware"]["device_map"], "max_memory": MODEL_CONFIG["hardware"]["max_memory"], "offload_config": MODEL_CONFIG["hardware"]["offload_config"]}

        return {
            "status": "healthy",
            "timestamp": int(time.time()),
            "model": {
                "name": MODEL_CONFIG["model_name"],
                "loaded": model_manager.is_loaded,
                "info": model_info,
                "context_window": MODEL_CONFIG["max_sequence_length"],
                "max_tokens": MODEL_CONFIG["attention"]["max_new_tokens"],
                "code_specific": {
                    "languages": ["python", "javascript", "typescript", "html", "css", "sql"],
                    "code_completion": MODEL_CONFIG["code_specific"]["enable_code_completion"],
                    "line_numbers": MODEL_CONFIG["code_specific"]["enable_line_numbers"],
                    "preserve_indentation": MODEL_CONFIG["code_specific"]["preserve_indentation"],
                },
            },
            "memory": {**memory_info, "threshold_gb": API_CONFIG["memory_threshold_gb"], "hardware": hardware_info},
            "server": {
                "api_version": "v1",
                "workers": API_CONFIG["workers"],
                "timeout": API_CONFIG["timeout"],
                "rate_limit": API_CONFIG["rate_limit"],
                "code_features": {
                    "syntax_highlighting": MODEL_CONFIG["code_specific"]["syntax_highlighting"],
                    "line_numbers": MODEL_CONFIG["code_specific"]["enable_line_numbers"],
                    "code_completion": MODEL_CONFIG["code_specific"]["enable_code_completion"],
                    "streaming": True,
                },
            },
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


def setup_routes(app: FastAPI):
    """Setup enhanced routes with improved error handling"""

    # Include router
    app.include_router(router, prefix=API_CONFIG.get("prefix", ""), tags=["DeepSeek API"])

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "type": "api_error", "code": exc.status_code, "request_id": getattr(request.state, "request_id", None)}},
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": {"message": "Internal server error", "type": "server_error", "code": 500, "request_id": getattr(request.state, "request_id", None)}})

    # Add request tracking middleware
    @app.middleware("http")
    async def track_request(request: Request, call_next):
        request.state.request_id = f"req_{int(time.time() * 1000)}"
        request.state.start_time = time.time()

        response = await call_next(request)

        # Add timing headers
        process_time = time.time() - request.state.start_time
        response.headers.update({"X-Request-ID": request.state.request_id, "X-Process-Time": f"{process_time:.3f}"})

        return response

    logger.info(f"Routes configured - Endpoints: {list(API_CONFIG['endpoints'].values())}, Rate Limit: {API_CONFIG['rate_limit']['requests_per_minute']} rpm")
