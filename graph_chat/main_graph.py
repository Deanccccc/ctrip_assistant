"""
    主图定义
"""
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage

from graph_chat.assistant_node import create_assistant_node, UserContext, safe_tools, sensitive_tools, \
    sensitive_tool_names
from graph_chat.state import State
from tools.init_db import update_dates

import warnings
from langchain_core._api import LangChainBetaWarning
warnings.filterwarnings("ignore", category=LangChainBetaWarning)
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


# 条件边路由函数
def route_conditional_tools(state: State):
    """
    条件路由函数: assistant -> [safe_tools, sensitive_tools, END]
    :param state: 图全局状态
    :return:
    """
    # 获取下一个节点
    next_node = tools_condition(state)
    if next_node == END:
        return END

    # 判断是否需要中断【敏感工具调用需要人工确认】
    message = state["messages"][-1]
    tool_calls = message.tool_calls[0]
    if tool_calls["name"] in sensitive_tool_names:
        return "sensitive_tools"
    return "safe_tools"


# =========== 1、主图构建 ==========
builder = StateGraph(State)
# 注册节点
# builder.add_node("get_user_info", get_user_info)
builder.add_node("assistant", create_assistant_node())
builder.add_node("safe_tools", ToolNode(safe_tools))
builder.add_node("sensitive_tools", ToolNode(sensitive_tools))
# 构建边
# builder.add_edge(START, "get_user_info")
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    route_conditional_tools,
    ["safe_tools", "sensitive_tools", END]
)
builder.add_edge("safe_tools", "assistant")
builder.add_edge("sensitive_tools", "assistant")


# 编译图, 设置短期记忆, 在敏感工具调用前发生中断
graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["sensitive_tools"])
# draw_graph(graph, "graph4.png")


session_id = str(uuid.uuid4())  # 生成唯一session_id
update_dates()  # 更新时间为当前时间


config = {"configurable": {"thread_id": session_id,}}
context = UserContext(passenger_id="3442 587242")




# =========== 2、执行工作流 ==========
# 执行一轮图调用
def run(user_input):
    """
    执行一轮图调用。
    :param user_input: 用户输入
    """
    # V3流式调用
    stream = graph.stream_events(
        input=user_input,
        config=config,
        context=context,
        version="v3",
    )

    # 流式打印输出结果
    print('[智能助手回复]: ')
    for handle in stream.subgraphs:
        for message in handle.messages:
            for token in message.text:
                print(token, end="", flush=True)
    print()


# 敏感工具中断处理
def handle_interrupt():
    """
    中断处理: 敏感工具被 interrupt_before 拦下后, 进入人工审批循环。
    - 批准: input=None 从断点续跑, sensitive_tools 真正执行
    - 拒绝: 手工补 ToolMessage(as_node="sensitive_tools") 后再续跑, 避免脏历史导致下一轮 400
    """
    state = graph.get_state(config)

    while state.next:  # 有挂起节点 = 图被中断
        # 静态 interrupt_before 的中断没有 value, 待审批的工具调用要从最后一条 AIMessage 取
        pending = state.values["messages"][-1]
        calls = list(pending.tool_calls)
        if not calls:
            break

        print("\n[系统] 检测到敏感操作, 等待人工审批:")
        for tool_call in calls:
            print(f"  - 工具: {tool_call['name']}")
            print(f"    参数: {tool_call['args']}")

        approve = input('[审批] 批准执行吗? (y/n): ').strip().lower()
        if approve in ('y', 'yes'):
            run(None)  # 放行: 从断点续跑, sensitive_tools 节点真正执行
        else:
            graph.update_state(
                config,
                {"messages": [
                    ToolMessage(
                        content="用户拒绝了该操作，未执行。",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                    for tool_call in calls
                ]},
                as_node="sensitive_tools",
            )
            run(None)  # 从"敏感节点已处理完"的位置续跑, assistant 会收到拒绝消息并回应用户

        state = graph.get_state(config)  # 重新检查是否又有新的中断


# config传递thread_id, context传递上下文
while True:
    question = input('[用户]: ')
    if question.lower() in ['q', 'exit', 'quit']:
        print('对话结束，拜拜！')
        break

    run({"messages": [{"role": "user", "content": question}]})
    handle_interrupt()  # 每轮结束后检查并处理中断

