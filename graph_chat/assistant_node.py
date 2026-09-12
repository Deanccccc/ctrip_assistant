"""
    创建图的子节点 - AssistantNode(本质就是子图、子Agent)
"""
from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

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


# 动态提示词, 从上下文中动态注入user_info
@dynamic_prompt
def prompt(request: ModelRequest) -> str:
    # 从上下文中得到
    user_info = request.runtime.context.passenger_id
    now = datetime.now()

    return (
        "您是携程瑞士航空公司的客户服务助理。优先使用提供的工具搜索航班、公司政策和其他信息来帮助用户的查询。"
        "搜索时，请坚持不懈。如果第一次搜索没有结果，扩大您的查询范围。"
        "如果搜索为空，在放弃之前扩展您的搜索。\n\n"
        f"当前用户:\n<User>\n{user_info}\n</User>\n"
        f"当前时间: {now}."
    )


# 创建子图实例(子Agent)
def create_assistant_node():
    return create_agent(
        model=deepseek,   # 模型
        tools=tools,      # 工具列表
        middleware=[prompt],   # 中间件
        context_schema=UserContext,  # 用户上下文
    )