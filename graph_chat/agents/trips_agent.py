"""
    旅游景点助手: 景点/短途旅行推荐/预订/修改/取消
"""
from graph_chat.agents.base import create_domain_agent
from tools.trip_tools import (
    search_trip_recommendations, book_excursion, update_excursion, cancel_excursion,
)

TRIPS_PROMPT = """
你是携程的旅游景点客服专家, 只处理旅游景点/短途旅行业务:
- 按位置/名称/关键词搜索旅行推荐 (search_trip_recommendations)
- 预订 (book_excursion)、修改详情 (update_excursion)、取消预订 (cancel_excursion)
搜索时如果第一次没有结果, 扩大范围再试。
不属于旅游业务的问题不要处理, 简短说明后结束(主管会转接其他专家)。
"""

trips_agent = create_domain_agent(
    domain_prompt=TRIPS_PROMPT,
    safe_tools=[search_trip_recommendations],
    sensitive_tools=[book_excursion, update_excursion, cancel_excursion],
)
