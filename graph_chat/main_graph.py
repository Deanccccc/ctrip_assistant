"""
    主图定义
"""
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from graph_chat.assistant_node import create_assistant_node, UserContext, tools
from graph_chat.draw_png import draw_graph
from graph_chat.state import State
from tools.init_db import update_dates

import warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")



# =========== 1、主图构建 ==========
builder = StateGraph(State)

builder.add_node("assistant", create_assistant_node())
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition
)
builder.add_edge("tools", "assistant")

graph = builder.compile(checkpointer=MemorySaver())
draw_graph(graph, "graph2.png")


session_id = str(uuid.uuid4())  # 生成唯一session_id
update_dates()  # 更新时间为当前时间


config = {"configurable": {"thread_id": session_id,}}
context = UserContext(passenger_id="3442 587242")


# =========== 2、执行工作流 ==========
# config传递thread_id, context传递上下文
while True:
    question = input('用户：')
    if question.lower() in ['q', 'exit', 'quit']:
        print('对话结束，拜拜！')
        break

    events = graph.stream(
        input={"messages": [{"role": "user", "content": question}]},
        config=config,
        context=context,
        stream_mode="messages",
    )

    for chunk, metadata in events:
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()
