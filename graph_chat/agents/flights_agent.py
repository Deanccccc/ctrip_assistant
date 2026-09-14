"""
    航班助手: 机票查询/搜索/改签/退票
"""
from graph_chat.agents.base import create_domain_agent
from tools.flights_tools import (
    fetch_user_flight_information, search_flights,
    update_ticket_to_new_flight, cancel_ticket,
)
from tools.retriever_vector import lookup_policy

FLIGHTS_PROMPT = """
你是携程的航班客服专家, 只处理机票相关业务:
- 查询用户名下的机票信息 (fetch_user_flight_information)
- 按条件搜索航班 (search_flights)
- 机票改签 (update_ticket_to_new_flight) 和退票 (cancel_ticket)
改签/退票等写操作前, 先用 lookup_policy 查询航空公司政策确认是否允许。
如果第一次搜索没有结果, 扩大范围再试。
不属于机票业务的问题不要处理, 简短说明后结束(主管会转接其他专家)。
"""

flights_agent = create_domain_agent(
    domain_prompt=FLIGHTS_PROMPT,
    safe_tools=[fetch_user_flight_information, search_flights, lookup_policy],
    sensitive_tools=[update_ticket_to_new_flight, cancel_ticket],
)
