"""
    创建图的子节点 - AssistantNode(本质就是子图、子Agent)

    使用子图实现节点, 更加灵活
"""
from datetime import datetime

from langchain_core.messages import SystemMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolRuntime

from graph_chat.state import State
from graph_chat.user_context import UserContext
from llm import deepseek
from tools.car_tools import search_car_rentals, book_car_rental, update_car_rental, cancel_car_rental
from tools.flights_tools import fetch_user_flight_information, search_flights, update_ticket_to_new_flight, cancel_ticket
from tools.hotels_tools import search_hotels, book_hotel, update_hotel, cancel_hotel
from tools.retriever_vector import lookup_policy
from tools.trip_tools import search_trip_recommendations, book_excursion, update_excursion, cancel_excursion


# 工具列表
tools = [
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    update_ticket_to_new_flight,
    cancel_ticket,
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
]

deepseek_with_tools = deepseek.bind_tools(tools)


# ============ 动态 system prompt ============
def build_system_prompt(user_info: str) -> str:
    now = datetime.now()
    return (
        "您是携程瑞士航空公司的客户服务助理。优先使用提供的工具搜索航班、公司政策和其他信息来帮助用户的查询。"
        "搜索时，请坚持不懈。如果第一次搜索没有结果，扩大您的查询范围。"
        "如果搜索为空，在放弃之前扩展您的搜索。\n\n"
        f"当前用户:\n<User>\n{user_info}\n</User>\n"
        f"当前时间: {now}."
    )


# ============ LLM 决策节点 ============
def llm_node(state: State, runtime: ToolRuntime[UserContext, None]) -> dict:
    """
    从 runtime.context 读取 UserContext，动态注入 system prompt，
    调用 LLM，返回一条 AI 消息（可能带 tool_calls）。
    """
    # 从运行时上下文拿到用户信息（由主图调用子图时通过 context 传入）
    user_info = runtime.context.passenger_id

    # 组装消息：system message + 历史 messages
    system_msg = SystemMessage(content=build_system_prompt(user_info))
    llm_input = [system_msg] + state["messages"]

    # deepseek 必须是绑定了 tools 的模型，否则不会产生 tool_calls
    llm_response = deepseek_with_tools.invoke(llm_input)

    # 只返回新产生的 AI 消息，add_messages 会追加到 messages
    return {"messages": [llm_response]}


# ============ 构建手动子图 ============
def create_assistant_node():
    builder = StateGraph(State, context_schema=UserContext)

    builder.add_node("llm", llm_node)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)

    return builder.compile()