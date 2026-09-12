"""
    主图定义
"""

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from graph_chat.assistant_node import create_assistant_node, UserContext
from graph_chat.state import State


# 创建图
builder = StateGraph(State)

# 添加节点
builder.add_node("assistant", create_assistant_node())

# 添加边
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

# 编译图
graph = builder.compile()


response = graph.invoke(
    input={
        "messages": [
            {"role": "user", "content": "查询我的航班信息"}
        ],
    },
    config={
        "configurable": {
            "thread_id": "session-001",
        }
    },
    context=UserContext(passenger_id="3442 587242"),
)

print(response["messages"][-1].content)
