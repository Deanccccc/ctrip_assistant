"""
    主图定义: Supervisor + 4个领域子Agent 的多Agent协作(手动实现)
    结构: START -> supervisor -> (flights|hotels|cars|trips 子图) -> supervisor -> ... -> END
    敏感工具中断在子图内部触发, 自动冒泡到主图, 由主循环 handle_interrupt 统一审批
"""
import re
import uuid

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from graph_chat.draw_png import draw_graph
from graph_chat.supervisor import supervisor_node
from graph_chat.agents.flights_agent import flights_agent
from graph_chat.agents.hotels_agent import hotels_agent
from graph_chat.agents.cars_agent import cars_agent
from graph_chat.agents.trips_agent import trips_agent
from graph_chat.state import State
from graph_chat.user_context import UserContext
from tools.init_db import update_dates

import warnings
from langchain_core._api import LangChainBetaWarning
warnings.filterwarnings("ignore", category=LangChainBetaWarning)
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")



# =========== 1、主图构建 ==========
AGENT_NODES = ["flights_agent", "hotels_agent", "cars_agent", "trips_agent"]

builder = StateGraph(State, context_schema=UserContext)

# 节点注册, 一个主管节点 + 4个子Agent节点
builder.add_node("supervisor", supervisor_node)
builder.add_node("flights_agent", flights_agent)
builder.add_node("hotels_agent", hotels_agent)
builder.add_node("cars_agent", cars_agent)
builder.add_node("trips_agent", trips_agent)

# 添加边
builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],   # 条件函数, 动态添加边
    AGENT_NODES + [END],
)
# 专家干完活必须回主管复命, 由 supervisor 决定继续派单还是收尾
for name in AGENT_NODES:
    builder.add_edge(name, "supervisor")

# 编译图: 敏感工具中断在子图内部设置(interrupt_before), 会自动冒泡到主图
graph = builder.compile(checkpointer=MemorySaver())
draw_graph(graph, "graph6.png")


session_id = str(uuid.uuid4())  # 生成唯一session_id
update_dates()  # 更新时间为当前时间


config = {
    "configurable": {"thread_id": session_id},
    "recursion_limit": 50,  # 多跳路由的硬保险, 防止 supervisor 空转死循环
}
context = UserContext(passenger_id="3442 587242")


# =========== 2、执行工作流 ==========
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)  # 过滤深度思考过程(成对标签)


def _strip_think(text: str) -> str:
    """过滤模型的深度思考过程。
    v1: 删除成对的 <think>...</think>;
    v2: 兜底处理缺失开标签的情况 —— 保留最后一个 </think> 之后的内容。"""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.lstrip()


def run(user_input):
    """
    执行一轮图调用(极简打印版: 不做流式输出, 轮次结束后从最终 state 打印本轮新增的 AI 消息)。
    :param user_input: 正常轮次传 {"messages": [...]}; 传 None 表示从 interrupt 断点续跑
    """
    before = len(graph.get_state(config).values.get("messages", []))
    graph.invoke(input=user_input, config=config, context=context)

    print('[智能助手回复]: ')
    for message in graph.get_state(config).values.get("messages", [])[before:]:
        if message.type == "ai" and message.text:
            print(_strip_think(message.text), end="", flush=True)
    print()


def handle_interrupt():
    """
    中断处理(子图版): 敏感工具在子图内部被 interrupt_before 拦下后, 中断冒泡到主图。
    - 待审批的 tool_calls 要从 get_state(config, subgraphs=True) 的子图状态里取
    - 批准: input=None 从主图续跑, 子图任务自动恢复
    - 拒绝: 用子图的 namespaced config 补 ToolMessage(as_node="sensitive_tools") 再续跑,
            避免脏历史导致下一轮 400 insufficient tool messages
    """
    root = graph.get_state(config, subgraphs=True)
    while root.next:
        # 根层 next 形如 ('flights_agent', ...), 被中断的子图状态挂在 task.state 上
        sub_state = None
        for task in root.tasks:
            st = getattr(task, "state", None)
            if st and st.next and "sensitive_tools" in st.next:
                sub_state = st
                break
        if sub_state is None:
            break

        # 静态 interrupt_before 的中断没有 value, 待审批工具调用从子图最后一条 AIMessage 取
        pending = sub_state.values["messages"][-1]
        calls = list(pending.tool_calls)
        if not calls:
            break

        print("\n[系统] 检测到敏感操作, 等待人工审批:")
        for tc in calls:
            print(f"  - 工具: {tc['name']}")
            print(f"    参数: {tc['args']}")

        approve = input('[审批] 批准执行吗? (y/n): ').strip().lower()
        if approve in ('y', 'yes'):
            run(None)  # 放行: 从主图断点续跑, 子图从 sensitive_tools 断点继续
        else:
            # 拒绝: 在子图命名空间补 ToolMessage(as_node="sensitive_tools")
            graph.update_state(
                sub_state.config,
                {"messages": [
                    ToolMessage(
                        content="用户拒绝了该操作，未执行。",
                        name=tc["name"],
                        tool_call_id=tc["id"],
                    )
                    for tc in calls
                ]},
                as_node="sensitive_tools",
            )
            run(None)  # 续跑: 专家会收到拒绝消息并向用户解释

        root = graph.get_state(config, subgraphs=True)  # 重新检查是否又有新中断


# config传递thread_id, context传递上下文
if __name__ == "__main__":
    while True:
        question = input('[用户]: ')
        if question.lower() in ['q', 'exit', 'quit']:
            print('对话结束，拜拜！')
            break

        run({"messages": [{"role": "user", "content": question}]})
        handle_interrupt()  # 每轮结束后检查并处理中断(可能发生在任一子图内部)
