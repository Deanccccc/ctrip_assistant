"""
    Supervisor 主Agent: 意图识别 + 任务分派 + 复命再决策 + 终止判断
    手动实现: 普通节点 + 结构化输出, 不用任何封装库
"""
from typing import TypedDict, Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from env_utils import DEEPSEEK_API_KEY
from graph_chat.state import State
from llm import supervisor_base

# 主Agent提示词
SUPERVISOR_PROMPT = """你是携程智能客服的主管助手。你不处理任何具体业务, 只负责分析对话并做出路由决策。

可用的去向:
- flights_agent: 机票业务(查询机票、搜索航班、改签、退票)
- hotels_agent: 酒店业务(搜索、预订、修改入住/退房日期、取消)
- cars_agent: 租车业务(搜索、预订、修改、取消)
- trips_agent: 旅游景点/短途旅行业务(搜索、预订、修改、取消)
- FINISH: 专家已答复完毕且用户没有新的业务诉求, 结束本轮对话

决策规则:
1. 每次只选择一个去向; instruction 里用一句话写清要该专家完成的具体任务, 不要替专家填具体参数。
2. 专家答复后, 如果该域任务已完成且用户没有其他诉求, 立即 FINISH, 不要重复分派同一个专家。
3. 用户需要跨域服务时, 依次分派给多个专家, 每次只转接一个。
4. 用户只是打招呼、闲聊、感谢, 直接 FINISH, 并在 final_reply 里给出简短友好的答复。
5. 只要本轮已有专家向用户作出答复, final_reply 必须填空字符串, 不要复述专家的话。
6. 如果专家声明用户的某个需求不属于其业务范围, 而该需求对应其他专家, 必须继续分派给对应专家,
   不要就此 FINISH; 直到所有诉求都被处理完才能 FINISH。
"""


# 路由决策
class Route(TypedDict):
    """
        supervisor 的结构化路由决策
    """
    next_agent: Literal["flights_agent", "hotels_agent", "cars_agent", "trips_agent", "FINISH"]   # 下一个节点
    instruction: str    #
    final_reply: str


supervisor_llm = supervisor_base.with_structured_output(Route, method="function_calling")


# 主管节点: 根据用户意图进行子Agent路由
def supervisor_node(state: State) -> dict:
    """
    主管节点:
        读取全量对话历史, 结构化输出历史决策
    结构化输出不会往 messages 里写任何 tool_calls,
    因此不会产生"AIMessage(tool_calls) 无 ToolMessage 应答"的脏历史(否则子Agent下一轮必400)。
    """
    # 使用决策模型 进行意图识别
    route: Route = supervisor_llm.invoke(
        [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    )
    # 根据结构化输出获得下一个节点
    next_node = "__end__" if route["next_agent"] == "FINISH" else route["next_agent"]

    # 保险丝处理
    update = {"next": next_node}
    if route["next_agent"] == "FINISH" and route.get("final_reply"):
        # 保险丝: 本轮最后一条已经是专家的 AI 答复时, 无视模型的 final_reply, 避免画蛇添足
        last = state["messages"][-1] if state["messages"] else None
        already_answered = isinstance(last, AIMessage) and last.content
        if not already_answered:
            # 仅打招呼/闲聊等无专家参与的场景才由主管补话
            update["messages"] = [AIMessage(content=route["final_reply"])]
    return update
