"""
Unit tests for gen_kwargs type verification.

Verifies that temperature and max_tokens are properly typed (float and int)
in the generated API request payload.
"""

import asyncio
from unittest.mock import patch

from core import APIConfig, complete


def test_gen_kwargs_types_in_request():
    """Verify that gen_kwargs are properly typed in request payload."""
    config = APIConfig(
        url="http://test.com/v1/chat/completions",
        model="test-model",
        api_key="test-key",
        gen_kwargs={
            "temperature": 0.7,
            "max_tokens": 100,
            "reasoning_effort": "medium",
            "logit_bias": {"42": -100, "1234": 50},
        },
    )

    captured_payload = None

    class MockResp:
        ok = True

        async def json(self):
            return {"choices": [{"message": {"content": "test response"}}]}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class MockContextManager:
        def __init__(self, json_payload):
            nonlocal captured_payload
            captured_payload = json_payload

        async def __aenter__(self):
            return MockResp()

        async def __aexit__(self, *args):
            pass

    def mock_post(url, json=None):
        return MockContextManager(json)

    with patch("core.aiohttp.ClientSession") as mock_session:
        mock_session.return_value.__aenter__.return_value.post = mock_post
        asyncio.run(complete(["test prompt"], config))

    # Verify payload was captured
    assert captured_payload is not None, "Request payload should be captured"

    # Verify temperature is a float
    assert "temperature" in captured_payload, "temperature should be in payload"
    assert isinstance(
        captured_payload["temperature"], float
    ), f"temperature should be float, got {type(captured_payload['temperature'])}"
    assert captured_payload["temperature"] == 0.7

    # Verify max_tokens is an int
    assert "max_tokens" in captured_payload, "max_tokens should be in payload"
    assert isinstance(
        captured_payload["max_tokens"], int
    ), f"max_tokens should be int, got {type(captured_payload['max_tokens'])}"
    assert captured_payload["max_tokens"] == 100

    # Verify reasoning_effort is a string
    assert (
        "reasoning_effort" in captured_payload
    ), "reasoning_effort should be in payload"
    assert isinstance(
        captured_payload["reasoning_effort"], str
    ), f"reasoning_effort should be str, got {type(captured_payload['reasoning_effort'])}"
    assert captured_payload["reasoning_effort"] == "medium"

    # Verify logit_bias is a dict
    assert "logit_bias" in captured_payload, "logit_bias should be in payload"
    assert isinstance(
        captured_payload["logit_bias"], dict
    ), f"logit_bias should be dict, got {type(captured_payload['logit_bias'])}"
    assert captured_payload["logit_bias"] == {"42": -100, "1234": 50}

    print(f"✓ temperature type: {type(captured_payload['temperature'])}")
    print(f"✓ max_tokens type: {type(captured_payload['max_tokens'])}")
    print(f"✓ reasoning_effort type: {type(captured_payload['reasoning_effort'])}")
    print(f"✓ logit_bias type: {type(captured_payload['logit_bias'])}")
    print(f"✓ Full payload: {captured_payload}")
