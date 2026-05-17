from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_GROQ_MODELS = (
    "openai/gpt-oss-120b", "openai/gpt-oss-20b",
    "qwen/qwen3-32b", "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile", "llama-3.1-8b-instant"
)

REASONING_EFFORT = {
    "openai/gpt-oss-20b": "low",
    "openai/gpt-oss-120b": "low",
    "qwen/qwen3-32b": "none"
}

REASONING_FORMAT = {
    "openai/gpt-oss-20b": "hidden",
    "openai/gpt-oss-120b": "hidden",
    "qwen/qwen3-32b": "hidden",
}

RATE_LIMIT_HEADERS = (
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    "retry-after",
)


@dataclass(frozen=True)
class CoachResponse:
    content: str | None
    model: str | None
    rate_limit_headers: dict[str, str]
    rate_limited_models: tuple[str, ...] = ()
    error: str | None = None
    usage: dict[str, Any] | None = None

    @property
    def allowance_exceeded(self) -> bool:
        return self.content is None and bool(self.rate_limited_models) and self.error is None


def build_coach_system_prompt(
    model_key: str | None = None,
    model_desc: str | None = None,
    hp_desc: dict[str, str] | None = None,
) -> str:
    prompt = (
        "You are ProbaCoach, the ML coach inside ProbaViz, an educational Streamlit app for "
        "exploring classifier decision boundaries, probabilities, and metrics. "
        "The user can select only two features at a time; do not suggest adding more.\n\n"

        "Your personality is witty, dry, and lightly sarcastic. Be playful, "
        "but never mean or discouraging. Use short jokes about model behavior, "
        "overfitting, probability calibration, or chaotic hyperparameters. "
        "Never mock the user. Never joke about sensitive topics.\n\n"

        "Give concise, practical coaching for the current app state. Return "
        "Markdown with 1-3 actionable things to try, grounded in the supplied "
        "metrics and selected configuration. Focus on the model, "
        "hyperparameters, preprocessing, and metrics on train and test subsets.\n\n"

        "Do not invent metrics, datasets, or results. Do not claim certainty. "
        "Prefer short paragraphs or bullet points over tables. Prioritize "
        "technical clarity and actionable guidance over humor.\n\n"

        "Keep responses compact, insightful, and fun."
    )
    if model_key is not None:
        prompt += f"\n\nSelected model: {model_key}"
    if model_desc is not None:
        prompt += f"\n\nModel documentation:\n{model_desc.strip()}"
    if hp_desc is not None:
        param_docs = "\n".join(
            f"- {name}: {description.strip()}"
            for name, description in sorted(hp_desc.items())
        )
        prompt += f"\n\nHyperparameter documentation:\n{param_docs}"
    return prompt


def build_coach_payload(
    *,
    dataset_kind: str,
    dataset_name: str,
    preprocessing: str,
    model_key: str,
    model_params: dict[str, Any],
    train_metrics: dict[str, pd.DataFrame],
    test_metrics: dict[str, pd.DataFrame],
    selected_features: Iterable[Any],
    train_size: float | None,
    split_random_state: int | None,
    target_classes: Iterable[Any],
    data_shape: tuple[int, int],
    synthetic_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "dataset": {
            "kind": dataset_kind,
            "name": dataset_name,
            "shape": {"rows": data_shape[0], "columns": data_shape[1]},
            "selected_features": _jsonable(list(selected_features)),
            "target_classes": _jsonable(list(target_classes)),
            "train_size": train_size,
            "split_random_state": split_random_state,
            "synthetic_params": _jsonable(synthetic_params),
        },
        "preprocessing": {"scaling": preprocessing},
        "model": {
            "name": model_key,
            "params": _jsonable(model_params),
        },
        "metrics": {
            "train": _metrics_to_records(train_metrics),
            "test": _metrics_to_records(test_metrics),
        },
    }


def payload_to_message(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_genai_coach(
    *,
    api_key: str,
    system_prompt: str,
    payload: dict[str, Any],
    models: Iterable[str] = DEFAULT_GROQ_MODELS,
    client: Any | None = None,
    max_tokens: int = 512,
) -> CoachResponse:
    if client is None:
        try:
            from groq import Groq
        except ImportError:
            return CoachResponse(
                content=None,
                model=None,
                rate_limit_headers={},
                error="The `groq` package is not installed.",
            )
        client = Groq(api_key=api_key, max_retries=0)

    rate_limited: list[str] = []
    last_headers: dict[str, str] = {}
    for model in models:
        try:
            response, headers = _create_chat_completion(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_message=payload_to_message(payload),
                max_tokens=max_tokens,
            )
            content = _extract_message_content(response)
            return CoachResponse(
                content=content,
                model=model,
                rate_limit_headers=headers,
                rate_limited_models=tuple(rate_limited),
                usage=log_chat_completion_usage(response, requested_model=model),
            )
        except Exception as exc:
            if _is_rate_limit_error(exc):
                rate_limited.append(model)
                last_headers = _extract_rate_limit_headers(getattr(exc, "response", None))
                continue
            return CoachResponse(
                content=None,
                model=model,
                rate_limit_headers=_extract_rate_limit_headers(getattr(exc, "response", None)),
                rate_limited_models=tuple(rate_limited),
                error=str(exc),
            )

    return CoachResponse(
        content=None,
        model=None,
        rate_limit_headers=last_headers,
        rate_limited_models=tuple(rate_limited),
    )


def _create_chat_completion(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
) -> tuple[Any, dict[str, str]]:
    completions = client.chat.completions
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    raw_client = getattr(completions, "with_raw_response", None)
    request_kwargs = _chat_completion_kwargs(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    if raw_client is not None:
        raw_response = raw_client.create(**request_kwargs)
        return raw_response.parse(), _extract_rate_limit_headers(raw_response)

    response = completions.create(**request_kwargs)
    return response, {}


def _chat_completion_kwargs(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    reasoning_effort = REASONING_EFFORT.get(model)
    if reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = reasoning_effort
    reasoning_format = REASONING_FORMAT.get(model)
    if reasoning_format is not None:
        request_kwargs["reasoning_format"] = reasoning_format
    return request_kwargs


def log_chat_completion_usage(
    response: Any,
    *,
    requested_model: str | None = None,
    model_key: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    row: dict[str, Any] = {}
    if model_key is not None:
        row["model_key"] = model_key
    if requested_model is not None:
        row["requested_groq_model"] = requested_model

    response_model = getattr(response, "model", None)
    if response_model is not None:
        row["response_groq_model"] = response_model

    if usage is not None:
        row.update(_usage_to_dict(usage))

    if metadata:
        row.update(_jsonable(metadata))
    return row


def _extract_message_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")
    return content or ""


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if isinstance(usage, dict):
        return {str(key): _jsonable(value) for key, value in usage.items()}

    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        if isinstance(dumped, dict):
            return {str(key): _jsonable(value) for key, value in dumped.items()}

    result: dict[str, Any] = {}
    for name in dir(usage):
        if name.startswith("_"):
            continue
        value = getattr(usage, name)
        if callable(value):
            continue
        if isinstance(value, (str, bool, int, float, type(None), dict, list, tuple)):
            result[name] = _jsonable(value)
    return result


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    return exc.__class__.__name__ == "RateLimitError"


def _extract_rate_limit_headers(response: Any) -> dict[str, str]:
    headers = getattr(response, "headers", response)
    if headers is None:
        return {}
    result: dict[str, str] = {}
    for header in RATE_LIMIT_HEADERS:
        value = _header_get(headers, header)
        if value is not None:
            result[header] = str(value)
    if hasattr(headers, "items"):
        for key, value in headers.items():
            normalized_key = str(key).lower()
            if normalized_key.startswith("x-ratelimit-"):
                result[normalized_key] = str(value)
    return result


def _header_get(headers: Any, key: str) -> Any:
    if hasattr(headers, "get"):
        return headers.get(key) or headers.get(key.lower()) or headers.get(key.upper())
    return None


def _metrics_to_records(metrics: dict[str, pd.DataFrame]) -> dict[str, list[dict[str, Any]]]:
    return {
        name: _jsonable(frame.round(4).to_dict(orient="records"))
        for name, frame in metrics.items()
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return round(value, 6) if np.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Index):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _format_rate_limit_headers(headers):
    labels = {
        "x-ratelimit-remaining-requests": "remaining requests",
        "x-ratelimit-remaining-tokens": "remaining tokens",
        "x-ratelimit-limit-requests": "request limit",
        "x-ratelimit-limit-tokens": "token limit",
        "x-ratelimit-reset-requests": "request reset",
        "x-ratelimit-reset-tokens": "token reset",
        "retry-after": "retry after",
    }
    parts = [
        f"{label}: {headers[key]}"
        for key, label in labels.items()
        if key in headers
    ]
    parts.extend(
        f"{key}: {value}"
        for key, value in sorted(headers.items())
        if key.startswith("x-ratelimit-") and key not in labels
    )
    return "; ".join(parts)
