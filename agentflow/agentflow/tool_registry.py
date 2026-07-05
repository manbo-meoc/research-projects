from typing import Any, Callable


class ToolRegistry:
    """
    AgentFlow 的工具注册中心。

    负责管理工具函数，并向 Planner 提供工具说明。
    """

    def __init__(self):
        self.tools: dict[str, Callable[..., Any]] = {}
        self.tool_descriptions: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """
        注册一个工具。
        """
        self.tools[name] = func
        self.tool_descriptions[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }

    def register_schema(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """
        Register only a tool schema.

        This is used when an external environment, such as tau2-bench, owns
        the actual tool execution. AgentFlow only needs the schema for planning.
        """
        self.tool_descriptions[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """
        返回所有工具的说明，供 Planner/LLM 使用。
        """
        return list(self.tool_descriptions.values())

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        根据工具名和参数真正执行工具。
        """
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")

        tool_func = self.tools[name]
        return tool_func(**arguments)
