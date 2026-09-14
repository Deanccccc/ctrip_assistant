"""
    graph 状态类
"""
from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class State(TypedDict):
    # 消息字段, 使用AnyMessage 接收 AI、Human、ToolMessage, 设置状态归并器为add_messages
    messages: Annotated[list[AnyMessage], add_messages]
    # 用户信息字段
    user_info: str