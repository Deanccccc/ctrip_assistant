"""
    租车助手: 租车搜索/预订/修改/取消
"""
from graph_chat.agents.base import create_domain_agent
from tools.car_tools import (
    search_car_rentals, book_car_rental, update_car_rental, cancel_car_rental,
)

CARS_PROMPT = """
你是携程的租车客服专家, 只处理租车相关业务:
- 按位置/公司名称搜索租车服务 (search_car_rentals)
- 预订 (book_car_rental)、修改起止日期 (update_car_rental)、取消预订 (cancel_car_rental)
搜索时如果第一次没有结果, 扩大范围再试。
不属于租车业务的问题不要处理, 简短说明后结束(主管会转接其他专家)。
"""

cars_agent = create_domain_agent(
    domain_prompt=CARS_PROMPT,
    safe_tools=[search_car_rentals],
    sensitive_tools=[book_car_rental, update_car_rental, cancel_car_rental],
)

