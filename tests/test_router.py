"""Tests for the RouterLLMProvider and ModelRoutingTools."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from prysm.brain.router import RouterLLMProvider, is_thinking_query
from prysm.config.settings import Settings
from prysm.models.interactions import BrainResponse, LLMMessage
from prysm.tools.model_tools import ModelRoutingTools


def test_is_thinking_query():
    # Thinking queries
    assert is_thinking_query([LLMMessage(role="user", content="Why does the sky appear blue during the day?")])
    assert is_thinking_query([LLMMessage(role="user", content="Explain the logic behind quantum computing.")])
    assert is_thinking_query([LLMMessage(role="user", content="What are the trade-offs between SQL and NoSQL?")])
    assert is_thinking_query([LLMMessage(role="user", content="Solve this riddle: I speak without a mouth. What am I?")])
    assert is_thinking_query([LLMMessage(role="user", content="[think] How should we optimize this cache?")])
    assert is_thinking_query([LLMMessage(role="user", content="Think carefully about the architectural implications.")])

    # Coding / action queries (stay on primary coder model)
    assert not is_thinking_query([LLMMessage(role="user", content="Write a python function to parse json.")])
    assert not is_thinking_query([LLMMessage(role="user", content="Open Chrome and launch spotify.")])
    assert not is_thinking_query([LLMMessage(role="user", content="Set volume to 50 percent.")])
    assert not is_thinking_query([LLMMessage(role="user", content="What is the current time?")])
    assert not is_thinking_query([LLMMessage(role="user", content="Hello PRYSM, how are you?")])
    assert not is_thinking_query([])


def test_router_mode_switching():
    settings = Settings(
        llm_mode="local",
        local_primary_model="qwen2.5-coder:14b",
        local_thinking_model="qwen3.5:9b",
        llm_model="gpt-4o",
    )
    router = RouterLLMProvider(settings)

    assert router.get_mode() == "local"
    assert "qwen2.5-coder:14b" in router.get_status_description()

    router.set_mode("api")
    assert router.get_mode() == "api"
    assert "gpt-4o" in router.get_status_description()

    router.set_mode("local")
    assert router.get_mode() == "local"

    with pytest.raises(ValueError):
        router.set_mode("unknown_mode")


@pytest.mark.asyncio
async def test_router_local_routing():
    settings = Settings(
        llm_mode="local",
        local_primary_model="qwen2.5-coder:14b",
        local_thinking_model="qwen3.5:9b",
    )
    router = RouterLLMProvider(settings)

    # Mock inner providers
    router.primary_provider.generate_response = AsyncMock(
        return_value=BrainResponse(text="Code response", tool_calls=[])
    )
    router.thinking_provider.generate_response = AsyncMock(
        return_value=BrainResponse(text="Thinking response", tool_calls=[])
    )

    # Test standard coding / general message -> routes to primary
    resp1 = await router.generate_response([
        LLMMessage(role="user", content="Write a python script to list files")
    ])
    assert resp1.text == "Code response"
    router.primary_provider.generate_response.assert_called_once()
    router.thinking_provider.generate_response.assert_not_called()

    # Reset mocks
    router.primary_provider.generate_response.reset_mock()
    router.thinking_provider.generate_response.reset_mock()

    # Test thinking question -> routes to thinking model
    resp2 = await router.generate_response([
        LLMMessage(role="user", content="Why does water expand when freezing? Think step by step.")
    ])
    assert resp2.text == "Thinking response"
    router.thinking_provider.generate_response.assert_called_once()
    router.primary_provider.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_router_api_routing():
    settings = Settings(
        llm_mode="api",
        llm_model="gpt-4o",
    )
    router = RouterLLMProvider(settings)

    router.api_provider.generate_response = AsyncMock(
        return_value=BrainResponse(text="API response", tool_calls=[])
    )

    resp = await router.generate_response([
        LLMMessage(role="user", content="Hello from cloud")
    ])
    assert resp.text == "API response"
    router.api_provider.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_router_fallback_local_to_api():
    settings = Settings(
        llm_mode="local",
        local_primary_model="qwen2.5-coder:14b",
        llm_model="gpt-4o",
    )
    router = RouterLLMProvider(settings)

    # Local fails (e.g. Ollama service down)
    router.primary_provider.generate_response = AsyncMock(
        side_effect=Exception("Ollama connection refused")
    )
    # Secondary API harness succeeds
    router.api_provider.generate_response = AsyncMock(
        return_value=BrainResponse(text="Fallback API response", tool_calls=[])
    )

    resp = await router.generate_response([
        LLMMessage(role="user", content="Help me with code")
    ])
    assert resp.text == "Fallback API response"
    router.primary_provider.generate_response.assert_called_once()
    router.api_provider.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_model_routing_tools():
    router = MagicMock()
    router.get_mode.return_value = "local"
    router.get_status_description.return_value = "Local Mode Active"
    router.last_used_model = "qwen2.5-coder:14b"

    tools = ModelRoutingTools(router)
    assert len(tools.get_schemas()) == 2

    # Test status
    status = await tools.execute("get_model_status", {})
    assert status["active_mode"] == "local"
    assert status["last_used_model"] == "qwen2.5-coder:14b"

    # Test switch
    switch_res = await tools.execute("switch_model_mode", {"mode": "api"})
    assert switch_res["status"] == "success"
    router.set_mode.assert_called_once_with("api")

