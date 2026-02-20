#!/usr/bin/env python3
"""
OpenAI-compatible Local VLM Server (Optimized for RTX GPU)
==========================================================

Endpoints:
- GET  /v1/models
- POST /v1/chat/completions  (supports image_url with data URI base64)

Optimizations:
- Full CUDA GPU utilization
- Efficient memory management
- Proper thinking model output handling
- Better image processing for handwriting/math

Designed for RTX 3050 (4GB VRAM) and similar GPUs.
"""

from __future__ import annotations

import base64
import gc
import io
import os
import re
import time
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from transformers import BitsAndBytesConfig
except Exception as e:
    raise RuntimeError(
        "Transformers is missing vision2seq/bnb support. Update transformers/bitsandbytes."
    ) from e

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local VLM Server (Optimized)", version="0.2")

# Configuration - optimized for RTX 3050 Laptop (4GB VRAM)
# Use official Qwen model from HuggingFace - will apply 4-bit quantization at runtime
DEFAULT_MODEL_ID = os.environ.get(
    "VLM_MODEL_ID", 
    "Qwen/Qwen2.5-VL-3B-Instruct"  # Official HF model, ~3B params fits in 4GB with quantization
)
DEFAULT_DEVICE = os.environ.get("VLM_DEVICE", "cuda")
DEFAULT_MAX_IMAGE_SIZE = int(os.environ.get("VLM_MAX_IMAGE_SIZE", "768"))  # Larger for better OCR

# Global model state
_model = None
_processor = None
_model_id = None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _strip_data_uri(uri: str) -> str:
    """Extract base64 from data URI."""
    m = re.match(r"^data:.*?;base64,(.*)$", uri, re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else uri


def _decode_image_from_data_uri(uri: str) -> Image.Image:
    """Decode base64 image from data URI."""
    b64 = _strip_data_uri(uri)
    try:
        data = base64.b64decode(b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image bytes: {e}")
    return img


def _resize_max(img: Image.Image, max_side: int) -> Image.Image:
    """Resize image to fit within max_side while preserving aspect ratio."""
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)  # High quality resize


def _is_bnb_quantized_model(model_id: str) -> bool:
    """Check if model is pre-quantized."""
    mid = (model_id or "").lower()
    return any(x in mid for x in ["bnb-2bit", "bnb-4bit", "2bit", "4bit", "bitsandbytes"])


def _clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


# ============================================================================
# MODEL LOADING (Optimized)
# ============================================================================

def _load_model(model_id: str):
    """Load model with optimal settings for RTX GPU."""
    global _model, _processor, _model_id

    if _model is not None and _processor is not None and _model_id == model_id:
        return

    logger.info(f"🔄 Loading model: {model_id}")
    load_start = time.time()
    
    _clear_gpu_memory()
    
    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        
        # Log GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        
        is_prequantized = _is_bnb_quantized_model(model_id)
        
        if is_prequantized:
            # Pre-quantized model - load directly
            logger.info("   Loading pre-quantized model (optimized)")
            _model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map={"": 0},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            # Apply 4-bit quantization for unquantized models
            logger.info("   Applying 4-bit quantization")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            _model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map={"": 0},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
    else:
        # CPU fallback
        logger.warning("⚠️ No GPU available, using CPU (slower)")
        _model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    _model.eval()
    _model_id = model_id
    
    # Report memory usage
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        logger.info(f"   VRAM: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
    
    load_time = time.time() - load_start
    logger.info(f"✅ Model loaded in {load_time:.2f}s")


# ============================================================================
# API MODELS
# ============================================================================

class ImageURL(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


class Message(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL_ID)
    messages: List[Message]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/v1/models")
def list_models():
    """List available models."""
    return {"data": [{"id": DEFAULT_MODEL_ID, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    """Handle chat completion requests with vision support."""
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported")

    model_id = req.model or DEFAULT_MODEL_ID
    _load_model(model_id)

    start_time = time.time()

    # Parse messages
    text_parts: List[str] = []
    images: List[Image.Image] = []
    system_prompt = ""

    for msg in req.messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_prompt = msg.content.strip()
            continue
            
        if msg.role != "user":
            continue

        if isinstance(msg.content, str):
            text_parts.append(msg.content)
        elif isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict):
                    ptype = part.get("type")
                    if ptype == "text":
                        text_parts.append(part.get("text", ""))
                    elif ptype == "image_url":
                        url = (part.get("image_url") or {}).get("url")
                        if not url:
                            raise HTTPException(status_code=400, detail="image_url.url missing")
                        img = _decode_image_from_data_uri(url)
                        img = _resize_max(img, DEFAULT_MAX_IMAGE_SIZE)
                        images.append(img)
                else:
                    cp = ContentPart.model_validate(part)
                    if cp.type == "text":
                        text_parts.append(cp.text or "")
                    elif cp.type == "image_url":
                        if not cp.image_url or not cp.image_url.url:
                            raise HTTPException(status_code=400, detail="image_url.url missing")
                        img = _decode_image_from_data_uri(cp.image_url.url)
                        img = _resize_max(img, DEFAULT_MAX_IMAGE_SIZE)
                        images.append(img)
        else:
            raise HTTPException(status_code=400, detail="Unsupported message.content")

    user_text = "\n\n".join([p for p in text_parts if p and str(p).strip()]).strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="No prompt text provided")

    # Create placeholder image if none provided
    if len(images) == 0:
        images = [Image.new("RGB", (32, 32), (255, 255, 255))]

    # Build conversation for Qwen3-VL
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Include image placeholder in content
    user_content = [
        {"type": "image"},
        {"type": "text", "text": user_text}
    ]
    messages.append({"role": "user", "content": user_content})
    
    # Apply chat template
    text_prompt = _processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Process inputs
    inputs = _processor(
        text=[text_prompt],
        images=images,
        return_tensors="pt",
        padding=True
    )

    # Move to GPU
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Log request details
    logger.info(f"📥 REQUEST: {len(images)} image(s) ({images[0].size if images else 'N/A'})")
    logger.info(f"   Prompt: {len(user_text)} chars, max_tokens: {req.max_tokens}")

    # Generate with optimizations
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            generated = _model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=req.temperature > 0,
                temperature=req.temperature if req.temperature > 0 else None,
                pad_token_id=_processor.tokenizer.pad_token_id if hasattr(_processor, 'tokenizer') else None,
            )

    # Decode only new tokens
    input_len = inputs["input_ids"].shape[1]
    out_text = _processor.batch_decode(
        generated[:, input_len:],
        skip_special_tokens=True
    )[0].strip()

    # Post-process thinking model output
    out_text = _clean_thinking_output(out_text)

    elapsed = time.time() - start_time
    
    # Log response
    logger.info(f"📤 RESPONSE: {len(out_text)} chars in {elapsed:.2f}s")
    logger.info(f"   Preview: {out_text[:150]}{'...' if len(out_text) > 150 else ''}")

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": out_text}, "finish_reason": "stop"}
        ],
    }


def _clean_thinking_output(text: str) -> str:
    """
    Clean up thinking model output to extract actual content.
    The thinking model may include reasoning before the answer.
    """
    # Check for explicit answer markers
    markers = [
        "**Answer:**", "**Output:**", "**Result:**",
        "Answer:", "Output:", "Result:",
        "The corrected text is:", "Here is the corrected text:",
        "Corrected text:", "Final answer:"
    ]
    
    for marker in markers:
        if marker.lower() in text.lower():
            idx = text.lower().find(marker.lower())
            text = text[idx + len(marker):].strip()
            break
    
    # Remove thinking blocks wrapped in <think> tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up markdown artifacts
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.split('\n')
        text = '\n'.join(lines[1:-1]).strip()
    
    return text


# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Pre-load model on startup."""
    logger.info("🚀 Starting VLM Server...")
    try:
        _load_model(DEFAULT_MODEL_ID)
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        # Don't raise - let server start anyway


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _model, _processor, _model_id
    logger.info("🛑 Shutting down VLM Server...")
    _model = None
    _processor = None
    _model_id = None
    _clear_gpu_memory()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("VLM_HOST", "0.0.0.0")
    port = int(os.environ.get("VLM_PORT", "8000"))
    
    logger.info(f"🌐 Starting server on {host}:{port}")
    uvicorn.run("vlm_server:app", host=host, port=port, reload=False, workers=1)
