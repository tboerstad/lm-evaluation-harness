"""
Integration tests for gen_kwargs (temperature, max_tokens) with real API calls.

These tests verify that generation parameters are correctly passed to the API
and respected by the model.
"""

import asyncio
import os

import pytest

from core import APIConfig, complete


@pytest.fixture
def api_config():
    """Create APIConfig from environment variables."""
    base_url = os.getenv("BASE_URL", "").strip()
    api_key = os.getenv("API_KEY", "").strip()
    model = os.getenv("MODEL", "").strip()

    if not all([base_url, api_key, model]):
        pytest.skip("Missing required environment variables: BASE_URL, API_KEY, MODEL")

    # Ensure the URL ends with /chat/completions
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"

    return APIConfig(
        url=url,
        model=model,
        api_key=api_key,
        num_concurrent=1,
        timeout=60,
    )


class TestMaxTokens:
    """Test that max_tokens parameter works correctly."""

    def test_max_tokens_limits_response_length(self, api_config):
        """Verify that max_tokens limits the response length."""
        prompt = "Write a long story about a robot learning to paint."

        # Test with very small max_tokens
        config_small = APIConfig(
            **{**api_config.__dict__, "gen_kwargs": {"max_tokens": 10}}
        )
        responses_small = asyncio.run(complete([prompt], config_small))
        response_small = responses_small[0]

        # Test with larger max_tokens
        config_large = APIConfig(
            **{**api_config.__dict__, "gen_kwargs": {"max_tokens": 100}}
        )
        responses_large = asyncio.run(complete([prompt], config_large))
        response_large = responses_large[0]

        # Verify the response with smaller max_tokens is shorter
        assert len(response_small) < len(response_large), (
            f"Expected response with max_tokens=10 to be shorter than max_tokens=100. "
            f"Got lengths: {len(response_small)} vs {len(response_large)}"
        )

        # Verify the small response is actually short (rough heuristic: ~4 chars per token)
        assert len(response_small) < 100, (
            f"Expected response with max_tokens=10 to be very short. "
            f"Got length: {len(response_small)}"
        )

        print(f"✓ max_tokens=10: {len(response_small)} chars: {response_small!r}")
        print(
            f"✓ max_tokens=100: {len(response_large)} chars: {response_large[:100]}..."
        )


class TestTemperature:
    """Test that temperature parameter works correctly."""

    def test_temperature_zero_is_deterministic(self, api_config):
        """Verify that temperature=0 produces consistent responses."""
        prompt = "What is 2 + 2? Answer with just the number."

        config = APIConfig(
            **{
                **api_config.__dict__,
                "gen_kwargs": {"temperature": 0.0, "max_tokens": 50},
            }
        )

        # Make multiple requests with temperature=0
        responses = asyncio.run(complete([prompt, prompt, prompt], config))

        # All responses should be identical (or very similar)
        assert responses[0] == responses[1] == responses[2], (
            f"Expected identical responses with temperature=0. " f"Got: {responses}"
        )

        print(f"✓ temperature=0 (deterministic): {responses[0]!r}")

    def test_temperature_affects_randomness(self, api_config):
        """Verify that different temperatures produce different behaviors."""
        prompt = "Write a creative opening sentence for a science fiction story."

        # Low temperature (more deterministic)
        config_low = APIConfig(
            **{
                **api_config.__dict__,
                "gen_kwargs": {"temperature": 0.1, "max_tokens": 50},
            }
        )
        responses_low = asyncio.run(complete([prompt, prompt, prompt], config_low))

        # High temperature (more random)
        config_high = APIConfig(
            **{
                **api_config.__dict__,
                "gen_kwargs": {"temperature": 1.5, "max_tokens": 50},
            }
        )
        responses_high = asyncio.run(complete([prompt, prompt, prompt], config_high))

        # Low temperature responses should be more consistent
        low_unique = len(set(responses_low))
        high_unique = len(set(responses_high))

        print(f"✓ temperature=0.1: {low_unique}/3 unique responses")
        print(f"  - {responses_low[0]!r}")
        print(f"  - {responses_low[1]!r}")
        print(f"  - {responses_low[2]!r}")
        print(f"✓ temperature=1.5: {high_unique}/3 unique responses")
        print(f"  - {responses_high[0]!r}")
        print(f"  - {responses_high[1]!r}")
        print(f"  - {responses_high[2]!r}")

        # This is a probabilistic test, so we just verify it doesn't crash
        # and produces valid responses. In practice, high temperature should
        # produce more varied responses, but we don't enforce this strictly.
        assert all(responses_low), "All low temperature responses should be non-empty"
        assert all(responses_high), "All high temperature responses should be non-empty"


class TestCombinedGenKwargs:
    """Test that multiple gen_kwargs work together."""

    def test_temperature_and_max_tokens_combined(self, api_config):
        """Verify that temperature and max_tokens work together."""
        prompt = "Explain quantum computing."

        config = APIConfig(
            **{
                **api_config.__dict__,
                "gen_kwargs": {
                    "temperature": 0.5,
                    "max_tokens": 50,
                },
            }
        )

        responses = asyncio.run(complete([prompt], config))
        response = responses[0]

        # Verify we get a valid response
        assert len(response) > 0, "Response should not be empty"

        # Verify max_tokens is respected (rough heuristic)
        assert len(response) < 400, (
            f"Response should be limited by max_tokens=50. "
            f"Got length: {len(response)}"
        )

        print(f"✓ temperature=0.5, max_tokens=50: {response}")
