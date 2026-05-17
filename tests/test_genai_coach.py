from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

from sklearn.datasets import load_iris
from sklearn.svm import SVC


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.genai_coach import (  # noqa
    DEFAULT_GROQ_MODELS,
    REASONING_EFFORT,
    REASONING_FORMAT,
    build_coach_payload,
    build_coach_system_prompt,
    log_chat_completion_usage,
    payload_to_message,
    request_genai_coach,
)
from src.viz import ProbaViz  # noqa


def _fitted_viz() -> ProbaViz:
    iris = load_iris(as_frame=True)
    data = iris["data"].iloc[:100].copy()
    target = iris["target"].iloc[:100].to_numpy()
    viz = ProbaViz(
        model=SVC(probability=True, C=0.5),
        data=data,
        target=target,
        train_size=0.7,
        split_random_state=42,
        features=[0, 1],
    )
    viz.fit()
    return viz


def test_build_coach_payload_is_json_serializable_and_contains_metrics():
    viz = _fitted_viz()
    payload = build_coach_payload(
        dataset_kind="Toy",
        dataset_name="Iris (multi-class)",
        preprocessing="None",
        model_key="Support Vector",
        model_params=viz.model.get_params(),
        train_metrics=viz.get_classification_metrics("train"),
        test_metrics=viz.get_classification_metrics("test"),
        selected_features=viz.features,
        train_size=viz.train_size,
        split_random_state=viz.split_random_state,
        target_classes=viz.classes,
        data_shape=viz.data.shape,
    )

    encoded = payload_to_message(payload)
    decoded = json.loads(encoded)

    assert decoded["dataset"]["kind"] == "Toy"
    assert decoded["dataset"]["train_size"] == 0.7
    assert decoded["model"]["params"]["C"] == 0.5
    assert decoded["metrics"]["train"]["aggregate_df"]
    assert decoded["metrics"]["test"]["class_specific_df"]


def test_system_prompt_is_stable_for_same_model_docs():
    prompt_1 = build_coach_system_prompt(
        "Support Vector",
        "Model docs",
        {"C": "Regularization", "kernel": "Kernel type"},
    )
    prompt_2 = build_coach_system_prompt(
        "Support Vector",
        "Model docs",
        {"kernel": "Kernel type", "C": "Regularization"},
    )

    assert prompt_1 == prompt_2
    assert "Support Vector" in prompt_1
    assert "Regularization" in prompt_1


def test_system_prompt_constrains_advice_to_app_state():
    prompt = build_coach_system_prompt()

    assert prompt.startswith("Formatting re-enabled\n/no_think")
    assert "JSON payload as the only source of truth" in prompt
    assert "exactly two selected features" in prompt
    assert "never suggest adding, removing, engineering, or selecting more features" in prompt
    assert "only hyperparameters present in model.params" in prompt
    assert "do not name missing tuning knobs" in prompt
    assert "oversampling" in prompt
    assert "cross-validation" in prompt
    assert "interpretation caveats" in prompt
    assert "1-3 actionable suggestions" in prompt


class _RawResponse:
    def __init__(self, content: str):
        self._content = content
        self.headers = {
            "x-ratelimit-remaining-requests": "10",
            "x-ratelimit-reset-tokens": "7.66s",
            "x-ratelimit-limit-tokens-day": "200000",
        }

    def parse(self):
        return SimpleNamespace(
            model="openai/gpt-oss-20b",
            usage=SimpleNamespace(
                prompt_tokens=123,
                completion_tokens=45,
                total_tokens=168,
            ),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._content)
                )
            ]
        )


class _RateLimitError(Exception):
    status_code = 429


class _RawCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _RawResponse(outcome)


class _Completions:
    def __init__(self, outcomes):
        self.with_raw_response = _RawCompletions(outcomes)


class _Client:
    def __init__(self, outcomes):
        self.chat = SimpleNamespace(completions=_Completions(outcomes))


def test_request_genai_coach_returns_first_successful_model():
    client = _Client(["Try C=0.5"])

    response = request_genai_coach(
        api_key="test-key",
        system_prompt="system",
        payload={"state": "payload"},
        models=["model-a", "model-b"],
        client=client,
    )

    assert response.content == "Try C=0.5"
    assert response.model == "model-a"
    assert response.usage["prompt_tokens"] == 123
    assert response.rate_limit_headers["x-ratelimit-remaining-requests"] == "10"
    assert response.rate_limit_headers["x-ratelimit-reset-tokens"] == "7.66s"
    assert response.rate_limit_headers["x-ratelimit-limit-tokens-day"] == "200000"
    call = client.chat.completions.with_raw_response.calls[0]
    assert call["max_tokens"] == 512
    assert "reasoning_effort" not in call
    assert "reasoning_format" not in call
    assert call["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": '{"state":"payload"}'},
    ]


def test_request_genai_coach_falls_back_after_rate_limit():
    client = _Client([_RateLimitError("limited"), "Second model works"])

    response = request_genai_coach(
        api_key="test-key",
        system_prompt="system",
        payload={"state": "payload"},
        models=["model-a", "model-b"],
        client=client,
    )

    assert response.content == "Second model works"
    assert response.model == "model-b"
    assert response.rate_limited_models == ("model-a",)


def test_request_genai_coach_reports_allowance_exceeded():
    client = _Client([_RateLimitError("limited"), _RateLimitError("limited")])

    response = request_genai_coach(
        api_key="test-key",
        system_prompt="system",
        payload={"state": "payload"},
        models=["model-a", "model-b"],
        client=client,
    )

    assert response.allowance_exceeded is True
    assert response.content is None
    assert response.rate_limited_models == ("model-a", "model-b")


def test_request_genai_coach_sends_reasoning_controls_for_supported_models():
    client = _Client(["Try C=0.5"])

    response = request_genai_coach(
        api_key="test-key",
        system_prompt="system",
        payload={"state": "payload"},
        models=["openai/gpt-oss-20b"],
        client=client,
    )

    assert response.content == "Try C=0.5"
    call = client.chat.completions.with_raw_response.calls[0]
    assert call["reasoning_effort"] == "low"
    assert call["reasoning_format"] == "hidden"


def test_default_models_only_include_supported_reasoning_efforts():
    assert "openai/gpt-oss-safeguard-20b" not in DEFAULT_GROQ_MODELS
    for model in DEFAULT_GROQ_MODELS:
        if model in REASONING_EFFORT:
            assert model in REASONING_FORMAT


def test_log_chat_completion_usage_flattens_usage_metadata():
    response = SimpleNamespace(
        model="openai/gpt-oss-120b",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
    )

    row = log_chat_completion_usage(
        response,
        requested_model="openai/gpt-oss-20b",
        model_key="Decision Tree",
        metadata={"ok": True},
    )

    assert row == {
        "model_key": "Decision Tree",
        "requested_groq_model": "openai/gpt-oss-20b",
        "response_groq_model": "openai/gpt-oss-120b",
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "ok": True,
    }
