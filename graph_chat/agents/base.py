"""
    领域子图创建工厂:
        统一通过Graph的方式创建llm节点和子Agent
        子图的中断会冒泡到主流程中, 由人工处理
"""
from datetime import datetime

from langchain_core.messages import SystemMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime

from graph_chat.state import State
from graph_chat.user_context import UserContext
from llm import deepseek

# 构建LLM节点
def make_llm_node(domain_prompt: str, all_tools: list):
    """生成子Agent的 llm 节点: 动态注入 域prompt + 用户上下文 + 当前时间"""
    llm_with_tools = deepseek.bind_tools(all_tools)

    def llm_node(state: State, runtime: Runtime[UserContext]) -> dict:
        system = SystemMessage(
            content=domain_prompt
            + f"\n\n当前用户: {runtime.context.passenger_id}\n当前时间: {datetime.now()}"
        )
        return {"messages": [llm_with_tools.invoke([system] + state["messages"])]}

    return llm_node


# 创建领域子图Agent
def create_domain_agent(domain_prompt: str, safe_tools: list, sensitive_tools: list):
    """
    创建一个领域助手子图。
    编译后的 StateGraph 可直接作为主图节点使用:
    - 调用时主图 state 会传入子图(共享 messages), 子图结束后输出合并回主图
    - 子图不单独设 checkpointer, 自动继承主图的(按 checkpoint_ns 隔离)
    """
    all_tools = list(safe_tools) + list(sensitive_tools)
    sensitive_names = {t.name for t in sensitive_tools}
    llm_node = make_llm_node(domain_prompt, all_tools)

    def route_conditional_tools(state: State):
        """安全/敏感工具分流; any() 修复了旧版只看第一个 tool_call 的混合调用问题"""
        if tools_condition(state) == END:
            return END
        if any(tc["name"] in sensitive_names for tc in state["messages"][-1].tool_calls):
            return "sensitive_tools"
        return "safe_tools"

    builder = StateGraph(State, context_schema=UserContext)
    builder.add_node("llm", llm_node)
    builder.add_node("safe_tools", ToolNode(safe_tools))
    builder.add_node("sensitive_tools", ToolNode(sensitive_tools))
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", route_conditional_tools,
                                  ["safe_tools", "sensitive_tools", END])
    builder.add_edge("safe_tools", "llm")
    builder.add_edge("sensitive_tools", "llm")
    return builder.compile(interrupt_before=["sensitive_tools"])
