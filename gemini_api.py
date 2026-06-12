from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_URL = os.getenv("GEMINI_API_URL", "https://api.univibe.cc/openai/v1/chat/completions")
DEFAULT_MODEL = "gemini-3-flash-preview"
api_key = ""
_DEFAULT_CONNECT_TIMEOUT = 10
_DEFAULT_READ_TIMEOUT = 300
_last_error = threading.local()


def request_gemini(
    img_urls: list[str] | tuple[str, ...] | None,
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_retry: int = 3,
    connect_timeout: int = _DEFAULT_CONNECT_TIMEOUT,
    read_timeout: int = _DEFAULT_READ_TIMEOUT,
    api_url: str = API_URL,
    api_key: str | None = None,
) -> tuple[str, str]:
    _clear_last_error()
    for _ in range(max_retry):
        try:
            request_body = build_payload(img_urls, prompt, model)
            response_body = do_post(api_url, _resolve_api_key(api_key), request_body, connect_timeout, read_timeout)
            _clear_last_error()
            return parse_response(response_body)
        except Exception as exc:
            _last_error.value = str(exc)
            print(f"request failed: {exc}")
    return "", ""


def request_gemini_messages(
    messages: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_retry: int = 3,
    connect_timeout: int = _DEFAULT_CONNECT_TIMEOUT,
    read_timeout: int = _DEFAULT_READ_TIMEOUT,
    api_url: str = API_URL,
    api_key: str | None = None,
) -> tuple[str, str]:
    _clear_last_error()
    for _ in range(max_retry):
        try:
            request_body = build_messages_payload(messages, model)
            response_body = do_post(api_url, _resolve_api_key(api_key), request_body, connect_timeout, read_timeout)
            _clear_last_error()
            return parse_response(response_body)
        except Exception as exc:
            _last_error.value = str(exc)
            print(f"request failed: {exc}")
    return "", ""


def get_last_error() -> str:
    return getattr(_last_error, "value", "") or ""


def build_payload(
    img_urls: list[str] | tuple[str, ...] | None,
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> str:
    user_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

    if img_urls:
        for img_url in img_urls:
            if img_url is None or not str(img_url).strip():
                continue
            user_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": to_image_data_url(str(img_url))},
                }
            )

    return json.dumps(
        {
            "model": normalize_model(model),
            "messages": [{"role": "user", "content": user_parts}],
        },
        ensure_ascii=False,
    )


def build_messages_payload(messages: list[dict[str, Any]], model: str = DEFAULT_MODEL) -> str:
    return json.dumps(
        {
            "model": normalize_model(model),
            "messages": messages,
        },
        ensure_ascii=False,
    )


def do_post(
    url: str,
    api_key_value: str,
    body: str,
    connect_timeout: int = _DEFAULT_CONNECT_TIMEOUT,
    read_timeout: int = _DEFAULT_READ_TIMEOUT,
) -> str:
    del connect_timeout
    request = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key_value}",
            "Content-Type": "application/json; charset=UTF-8",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=read_timeout) as response:
            response_body = response.read().decode("utf-8")
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {response_body}")
            return response_body
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc


def parse_response(response_body: str) -> tuple[str, str]:
    data = json.loads(response_body)
    choices = data.get("choices")
    if not choices:
        return "", ""

    message = choices[0].get("message", {})
    think = extract_content(message.get("reasoning_content"))
    result = extract_content(message.get("content"))
    return think, result


def extract_content(element: Any) -> str:
    if element is None:
        return ""
    if isinstance(element, str):
        return element
    if isinstance(element, list):
        texts = [item.get("text", "") for item in element if isinstance(item, dict) and item.get("text")]
        return "\n".join(texts)
    return json.dumps(element, ensure_ascii=False)


def to_image_data_url(image_input: str) -> str:
    trimmed = image_input.strip()
    if trimmed.startswith("data:image/"):
        return trimmed
    if trimmed.startswith(("http://", "https://")):
        return "data:image/jpeg;base64," + image_url_to_base64(trimmed)
    return f"data:{detect_mime_type(trimmed)};base64,{image_file_to_base64(trimmed)}"


def normalize_model(model: str | None) -> str:
    return model.strip() if model and model.strip() else DEFAULT_MODEL


def detect_mime_type(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "image/jpeg"


def image_url_to_base64(image_url: str) -> str:
    request = urllib.request.Request(image_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise OSError(f"failed to download image, HTTP {response.status}, URL: {image_url}")
            return base64.b64encode(response.read()).decode("ascii")
    except urllib.error.HTTPError as exc:
        raise OSError(f"failed to download image, HTTP {exc.code}, URL: {image_url}") from exc


def image_file_to_base64(file_path: str) -> str:
    return base64.b64encode(Path(file_path).read_bytes()).decode("ascii")




def _resolve_api_key(value: str | None) -> str:
    resolved = value or api_key or os.getenv("GEMINI_API_KEY")
    if not resolved:
        raise RuntimeError("missing API key: set GEMINI_API_KEY or pass api_key")
    return resolved


def _clear_last_error() -> None:
    if hasattr(_last_error, "value"):
        delattr(_last_error, "value")


if __name__ == "__main__":
    image_urls = ["https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"]
    thinking, answer = request_gemini(image_urls, "图中描绘的是什么景象？")
    print("think :", thinking)
    print("result:", answer)
