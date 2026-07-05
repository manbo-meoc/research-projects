#所有信息都存在这里（共享内存）
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):#它继承的是：TypedDict，给字典加类型信息
    """
    AgentFlow 的运行状态。

    在 LangGraph 中，State 会在不同 node 之间传递和更新。
    """

    # 用户、Agent、Tool 的对话历史
    messages: list[dict[str, Any]]

    # Planner 生成的工具调用请求，第一个planner输出的结果
    tool_calls: list[dict[str, Any]]

    # Tool 执行后的返回结果
    tool_results: list[dict[str, Any]]

    # 最终回复 第二个planner输出的结果
    final_answer: str

    # 当前运行步数，用来避免死循环
    step_count: int
