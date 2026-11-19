import base64
import time
import urllib.parse
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import requests
from google import genai
from google.genai import types
from PIL import Image

SYSTEM_INSTRUCTION = (
    "You are an intelligent, polite, and helpful AI assistant. "
    "Always reply in the same language the user uses (default to clear, natural English unless the user writes in another language). "
    "When an image is provided, analyze it thoroughly and give an accurate, detailed description or answer questions based on it."
)


def get_client(api_key: str) -> genai.Client:
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta"),
    )


def _build_contents(
    history_messages: List[Dict],
    current_text: str,
    current_image_path: Optional[str] = None,
) -> List[types.Content]:
    contents: List[types.Content] = []

    for msg in history_messages[-12:]:
        role = "user" if msg["role"] == "user" else "model"
        parts: List[types.Part] = []

        if msg.get("text"):
            parts.append(types.Part.from_text(text=msg["text"]))

        img_path = msg.get("image_path")
        if img_path and Path(img_path).exists() and msg["role"] == "user":
            img_bytes = Path(img_path).read_bytes()
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

        if parts:
            contents.append(types.Content(role=role, parts=parts))

    current_parts: List[types.Part] = []
    if current_text.strip():
        current_parts.append(types.Part.from_text(text=current_text.strip()))
    else:
        current_parts.append(
            types.Part.from_text(text="Please describe what is in this image in detail.")
        )

    if current_image_path and Path(current_image_path).exists():
        img_bytes = Path(current_image_path).read_bytes()
        current_parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

    contents.append(types.Content(role="user", parts=current_parts))
    return contents


def stream_chat_response(
    api_key: str,
    model_name: str,
    history_messages: List[Dict],
    user_text: str,
    image_path: Optional[str] = None,
) -> Generator[str, None, None]:
    """Streams responses and automatically falls back to backup models if 503/429 occurs."""
    client = get_client(api_key)
    contents = _build_contents(history_messages, user_text, image_path)

    fallback_models = [model_name] + [
        m for m in ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-4-hydrogen"]
        if m != model_name
    ]

    last_error = None
    for idx, current_model in enumerate(fallback_models):
        yielded_any = False
        try:
            response_stream = client.models.generate_content_stream(
                model=current_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.7,
                ),
            )
            for chunk in response_stream:
                if chunk.text:
                    yielded_any = True
                    yield chunk.text
            return
        except Exception as e:
            last_error = e
            err_str = str(e)
            is_temp_error = any(
                code in err_str for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")
            )
            if not yielded_any and is_temp_error and idx < len(fallback_models) - 1:
                time.sleep(1)
                continue
            raise

    if last_error:
        raise last_error


def _enhance_prompt_for_image(
    gemini_api_key: str,
    chat_model: str,
    prompt: str,
    reference_image_path: Optional[str] = None,
) -> str:
    """Converts and enhances the user's request into a detailed English image prompt."""
    if not gemini_api_key:
        return prompt

    for model_to_try in [chat_model, "gemini-3.5-flash-lite"]:
        try:
            client = get_client(gemini_api_key)
            instruction = (
                "Convert the user's image request into a concise, detailed English image-generation prompt "
                "(under 60 words). Include subject, style, lighting, and atmosphere. "
                "If a reference image is attached, incorporate its key visual elements with the requested changes. "
                "Output ONLY the final English prompt, nothing else."
            )
            parts: List[types.Part] = [
                types.Part.from_text(text=f"{instruction}\n\nUser request: {prompt}")
            ]
            if reference_image_path and Path(reference_image_path).exists():
                img_bytes = Path(reference_image_path).read_bytes()
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

            resp = client.models.generate_content(
                model=model_to_try,
                contents=[types.Content(role="user", parts=parts)],
            )
            if resp.text and resp.text.strip():
                return resp.text.strip()
        except Exception:
            continue
    return prompt


def generate_free_pollinations_image(
    prompt: str,
    pollinations_key: str = "",
) -> bytes:
    """Generates an image using Pollinations.ai."""
    encoded = urllib.parse.quote(prompt)
    if pollinations_key.strip():
        url = f"https://gen.pollinations.ai/image/{encoded}?width=1024&height=1024&model=flux&nologo=true"
        headers = {"Authorization": f"Bearer {pollinations_key.strip()}"}
    else:
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&model=flux&nologo=true"
        headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(url, headers=headers, timeout=90)
    content_type = resp.headers.get("content-type", "")

    if resp.status_code == 200 and content_type.startswith("image/"):
        return resp.content

    if resp.status_code in (401, 402):
        raise RuntimeError(
            "Pollinations free anonymous limit reached. Please select 'Hugging Face (Free FLUX.1)' in the sidebar "
            "or add a free Pollinations API key from enter.pollinations.ai."
        )
    raise RuntimeError(f"Pollinations API error ({resp.status_code}): {resp.text[:200]}")


def generate_huggingface_image(prompt: str, hf_token: str) -> bytes:
    """Generates an image using Hugging Face FLUX.1-schnell."""
    if not hf_token.strip():
        raise ValueError("Hugging Face Token is required. Please enter it in the sidebar settings.")

    url = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    headers = {
        "Authorization": f"Bearer {hf_token.strip()}",
        "Accept": "image/png",
    }
    resp = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=90)
    content_type = resp.headers.get("content-type", "")

    if resp.status_code == 200 and content_type.startswith("image/"):
        return resp.content

    raise RuntimeError(f"Hugging Face error ({resp.status_code}): {resp.text[:250]}")


def generate_gemini_paid_image(
    api_key: str,
    model_name: str,
    prompt: str,
    reference_image_path: Optional[str] = None,
) -> Tuple[str, Optional[bytes]]:
    """Generates or edits an image using Gemini (requires paid billing)."""
    client = get_client(api_key)
    contents: List = [prompt]
    if reference_image_path and Path(reference_image_path).exists():
        contents = [Image.open(reference_image_path), prompt]

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )

    reply_text = ""
    image_bytes: Optional[bytes] = None
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if getattr(part, "text", None):
                reply_text += part.text + "\n"
            elif getattr(part, "inline_data", None) and part.inline_data.data:
                raw_data = part.inline_data.data
                image_bytes = base64.b64decode(raw_data) if isinstance(raw_data, str) else raw_data

    return reply_text.strip(), image_bytes


def generate_ai_image(
    provider: str,
    gemini_api_key: str,
    chat_model: str,
    prompt: str,
    reference_image_path: Optional[str] = None,
    hf_token: str = "",
    pollinations_key: str = "",
    gemini_image_model: str = "gemini-nano-banana-2.1",
) -> Tuple[str, Optional[bytes]]:
    """Routes image generation to the selected provider."""
    if provider == "Gemini (Paid Billing)":
        return generate_gemini_paid_image(
            api_key=gemini_api_key,
            model_name=gemini_image_model,
            prompt=prompt,
            reference_image_path=reference_image_path,
        )

    english_prompt = _enhance_prompt_for_image(
        gemini_api_key=gemini_api_key,
        chat_model=chat_model,
        prompt=prompt,
        reference_image_path=reference_image_path,
    )

    if provider == "Hugging Face (Free FLUX.1)":
        img_bytes = generate_huggingface_image(english_prompt, hf_token)
    else:
        img_bytes = generate_free_pollinations_image(english_prompt, pollinations_key)

    return f"✨ **Generated Prompt:** *{english_prompt}*", img_bytes