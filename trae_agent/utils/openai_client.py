# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""OpenAI API client wrapper with tool integration."""

import os
import json
import random
import time
import openai
from openai.types.responses import EasyInputMessageParam, FunctionToolParam, ResponseFunctionToolCallParam, ResponseInputParam
from openai.types.responses.response_input_param import FunctionCallOutput
from typing import override

from ..tools.base import Tool, ToolCall, ToolResult
from ..utils.config import ModelParameters
from .base_client import BaseLLMClient
from .llm_basics import LLMMessage, LLMResponse, LLMUsage


class OpenAIClient(BaseLLMClient):
    """OpenAI client wrapper with tool schema generation."""

    def __init__(self, model_parameters: ModelParameters):
        super().__init__(model_parameters)

        if self.api_key == "":
            self.api_key: str = os.getenv("OPENAI_API_KEY", "")

        if self.api_key == "":
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY in environment variables or config file.")

        self.client: openai.OpenAI = openai.OpenAI(api_key=self.api_key)
        self.message_history: ResponseInputParam = []

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    @override
    def chat(self, messages: list[LLMMessage], model_parameters: ModelParameters, tools: list[Tool] | None = None, reuse_history: bool = True) -> LLMResponse:
        """Send chat messages to OpenAI using the standard Chat Completions API."""

        api_request_messages = []
        if reuse_history:
            api_request_messages.extend(self.message_history)
        
        openai_messages_for_call = self.parse_messages(messages)
        api_request_messages.extend(openai_messages_for_call)

        tool_schemas = None
        if tools:
            tool_schemas = [
                {"type": "function", "function": tool.json_definition()}
                for tool in tools
            ]

        response = None
        error_message = ""

        api_params = {
            "temperature": model_parameters.temperature,
            "top_p": model_parameters.top_p,
        }
        if model_parameters.model == "o3":
            # The 'o3' model supports max_completion_tokens instead of max_tokens
            api_params["max_completion_tokens"] = model_parameters.max_tokens
            # The 'o3' model requires a temperature of exactly 1
            api_params["temperature"] = 1.0
        else:
            api_params["max_tokens"] = model_parameters.max_tokens

        for i in range(model_parameters.max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=api_request_messages,
                    model=model_parameters.model,
                    tools=tool_schemas if tool_schemas else openai.NOT_GIVEN,
                    tool_choice="auto",
                    **api_params,
                )
                break
            except Exception as e:
                error_message += f"Error {i + 1}: {str(e)}\n"
                time.sleep(random.randint(3, 30))
                continue

        if response is None:
            raise ValueError(f"Failed to get response from OpenAI after max retries: {error_message}")

        response_message = response.choices[0].message
        content = response_message.content or ""

        self.message_history = api_request_messages + [response_message.model_dump(exclude_unset=True)]
        
        tool_calls: list[ToolCall] = []
        if response_message.tool_calls:
            for tc in response_message.tool_calls:
                tool_calls.append(ToolCall(
                    call_id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                    id=tc.id
                ))

        usage = None
        if response.usage:
            usage = LLMUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cache_read_input_tokens=0,
                reasoning_tokens=0
            )

        llm_response = LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
            tool_calls=tool_calls if tool_calls else None
        )

        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider="openai",
                model=model_parameters.model,
                tools=tools
            )

        return llm_response

    @override
    def supports_tool_calling(self, model_parameters: ModelParameters) -> bool:
        """Check if the current model supports tool calling."""

        if 'o1-mini' in model_parameters.model:
            return False

        tool_capable_models = [
            "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
            "gpt-4.1", "gpt-4.5",
            "o1", "o3", "o4"
        ]
        return any(model in model_parameters.model for model in tool_capable_models)

    def parse_messages(self, messages: list[LLMMessage]) -> ResponseInputParam:
        """Parse the messages to OpenAI format."""
        openai_messages: ResponseInputParam = []
        for msg in messages:
            if msg.tool_result:
                tool_result = msg.tool_result
                result_content = ""
                if tool_result.result:
                    result_content += tool_result.result
                if tool_result.error:
                    result_content += f"\\nError: {tool_result.error}"
                
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_result.call_id,
                    "content": result_content.strip()
                })
            elif msg.tool_call:
                openai_messages.append(self.parse_tool_call(msg.tool_call))
            else:
                if not msg.content:
                    raise ValueError("Message content is required")
                if msg.role == "system":
                    openai_messages.append({"role": "system", "content": msg.content})
                elif msg.role == "user":
                    openai_messages.append({"role": "user", "content": msg.content})
                elif msg.role == "assistant":
                    openai_messages.append({"role": "assistant", "content": msg.content})
                else:
                    raise ValueError(f"Invalid message role: {msg.role}")
        return openai_messages

    def parse_tool_call(self, tool_call: ToolCall) -> ResponseFunctionToolCallParam:
        """Parse the tool call from the LLM response."""
        return ResponseFunctionToolCallParam(
            call_id=tool_call.call_id,
            name=tool_call.name,
            arguments=json.dumps(tool_call.arguments),
            type="function_call",
        )

    def parse_tool_call_result(self, tool_call_result: ToolResult) -> FunctionCallOutput:
        """Parse the tool call result from the LLM response."""
        result: str = ""
        if tool_call_result.result:
            result = result + tool_call_result.result + "\n"
        if tool_call_result.error:
            result += tool_call_result.error
        result = result.strip()

        return FunctionCallOutput(
            call_id=tool_call_result.call_id,
            id=tool_call_result.id,
            output=result,
            type="function_call_output",
        )