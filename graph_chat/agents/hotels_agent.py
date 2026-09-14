"""
    酒店助手: 酒店搜索/预订/修改/取消
"""
from graph_chat.agents.base import create_domain_agent
from tools.hotels_tools import search_hotels, book_hotel, update_hotel, cancel_hotel

HOTELS_PROMPT = """
你是携程的酒店客服专家, 只处理酒店相关业务:
- 按位置/名称搜索酒店 (search_hotels)
- 预订酒店 (book_hotel)、修改入住/退房日期 (update_hotel)、取消预订 (cancel_hotel)
搜索时如果第一次没有结果, 扩大范围再试。
不属于酒店业务的问题不要处理, 简短说明后结束(主管会转接其他专家)。
"""

hotels_agent = create_domain_agent(
    domain_prompt=HOTELS_PROMPT,
    safe_tools=[search_hotels],
    sensitive_tools=[book_hotel, update_hotel, cancel_hotel],
)
