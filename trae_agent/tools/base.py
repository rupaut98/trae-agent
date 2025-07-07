# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Base classes for tools and tool calling."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import override


class ToolError(Exception):
    """Base class for tool errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message: str = message

@dataclass
class ToolExecResult:
    """Intermediate result of a tool execution."""
    output: str | None = None
    error: str | None = None
    error_code: int = 0

@dataclass
class ToolResult:
    """Result of a tool execution."""
    call_id: str
    success: bool
    result: str | None = None
    error: str | None = None
    id: str | None = None # OpenAI-specific field


ToolCallArguments = dict[str, str | int | float | dict[str, object] | list[object] | None]

@dataclass
class ToolCall:
    """Represents a parsed tool call."""
    name: str
    call_id: str
    arguments: ToolCallArguments = field(default_factory=dict)
    id: str | None = None

    @override
    def __str__(self) -> str:
        return f"ToolCall(name={self.name}, arguments={self.arguments}, call_id={self.call_id}, id={self.id})"


@dataclass
class ToolParameter():
    """Tool parameter definition."""
    name: str
    type: str | list[str]
    description: str
    enum: list[str] | None = None
    items: dict[str, object] | None = None
    required: bool = True


class Tool(ABC):
    """Base class for all tools."""

    def __init__(self, model_provider: str | None = None):
        self.model_provider = model_provider
        self.name: str = self.get_name()
        self.description: str = self.get_description()
        self.parameters: list[ToolParameter] = self.get_parameters()

    @abstractmethod
    def get_name(self) -> str:
        """Get the tool name."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get the tool description."""
        pass

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """Get the tool parameters."""
        pass

    @abstractmethod
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        """Execute the tool with given parameters."""
        pass

    def json_definition(self) -> dict[str, object]:
        return {
            "name": self.get_name(),
            "description": self.get_description(),
            "parameters": self.get_input_schema()
        }

    def get_input_schema(self) -> dict[str, object]:
        """Get the input schema for the tool."""
        schema: dict[str, object] = {
            "type": "object",
        }

        properties: dict[str, dict[str, str | list[str] | dict[str, object]]] = {}
        required: list[str] = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum

            if param.items:
                properties[param.name]["items"] = param.items

            if param.required:
                required.append(param.name)

        schema["properties"] = properties
        if len(required) > 0:
            schema["required"] = required
        
        # extra properties are not allowed
        schema["additionalProperties"] = False

        return schema


class ToolExecutor:
    """Tool executor that manages tool execution."""

    def __init__(self, tools: list[Tool]):
        self.tools: dict[str, Tool] = {tool.name: tool for tool in tools}

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        if tool_call.name not in self.tools:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_call.name}' not found. Available tools: {list(self.tools.keys())}",
                call_id=tool_call.call_id,
                id=tool_call.id
            )

        tool = self.tools[tool_call.name]

        try:
            tool_exec_result = await tool.execute(tool_call.arguments)
            return ToolResult(
                success=tool_exec_result.error_code == 0,
                result=tool_exec_result.output,
                error=tool_exec_result.error,
                call_id=tool_call.call_id,
                id=tool_call.id
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error executing tool '{tool_call.name}': {str(e)}",
                call_id=tool_call.call_id,
                id=tool_call.id
            )

    async def parallel_tool_call(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls in parallel"""
        return await asyncio.gather(*[self.execute_tool_call(call) for call in tool_calls])

    async def sequential_tool_call(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls in sequential"""
        return [await self.execute_tool_call(call) for call in tool_calls]
