# CtripAssistant 多Agent架构分析

> 版本基准：2026-09-14 的多Agent协作实现（Supervisor 手写模式）
> 适用代码：`graph_chat/supervisor.py`、`graph_chat/agents/*`、`graph_chat/main_graph.py`、`graph_chat/state.py`

---

## 1. 模式定位

当前实现属于多Agent设计地图中的 **BASELINE：Subagent（聚焦委派）** 模式：

- **控制权**：主 Agent（supervisor）逐轮决定下一步去哪，无预设路径
- **状态**：结果回到调用方（子图输出合并回主图 State），专家不持有跨轮私有状态
- **通信**：专家只与 supervisor 双向通信，**专家之间零直接通信**
- **复用**：`create_domain_agent()` 工厂复用的是 Agent 定义（人设 + 工具集），编排结构四域完全一致

一个常被混淆的点：主图节点数是 **1 个 supervisor + 4 个子图节点**，而不是"5 个助手节点平铺"——4 个助手的 LLM 节点和工具节点都封装在各自子图内部。

## 2. 整体结构

```
graph_chat/
├── main_graph.py        # 主图: supervisor + 4子图节点 + 条件边/回环边 + 主循环(中断审批)
├── supervisor.py        # 主Agent: 结构化输出路由
├── agents/
│   ├── base.py          # ★ 工厂: create_domain_agent() 四域共用的子图模板
│   ├── flights_agent.py # 航班: 查票/搜航班/改签/退票 (+lookup_policy 政策RAG)
│   ├── hotels_agent.py  # 酒店: 搜索/预订/修改/取消
│   ├── cars_agent.py    # 租车: 搜索/预订/修改/取消
│   └── trips_agent.py   # 景点: 推荐/预订/修改/取消
├── state.py             # State: messages + user_info + next(路由字段)
└── user_context.py      # UserContext(passenger_id), 全层贯穿
```

主图拓扑：

```
START ──> supervisor ──┬──> flights_agent ──┐
                       ├──> hotels_agent ───┤
                       ├──> cars_agent ─────┼──> 回 supervisor (复命)
                       ├──> trips_agent ────┘
                       └──> END (FINISH)
```

- 每个子图节点内部自带 `llm → 条件路由 → safe_tools / sensitive_tools → 回 llm` 循环
- 子图编译时设置 `interrupt_before=["sensitive_tools"]`，中断自动冒泡到主图
- 主图编译只挂一个 `MemorySaver`，子图状态按 `checkpoint_ns` 命名空间自动隔离

## 3. 核心组件

### 3.1 State（graph_chat/state.py）

```python
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 全量对话历史, 所有节点追加共享
    user_info: str
    next: str        # supervisor 路由决策: "flights_agent"/.../"__end__"
```

`messages` 是全局唯一的一条时间线——supervisor 与所有专家读写同一份历史，这是 Subagent 模式"结果回到调用方"的落地方式。代价是 token 随轮次增长，二阶段可演进为"supervisor 传任务摘要 + 子图私有消息通道"。

### 3.2 supervisor（graph_chat/supervisor.py）

- 独立低温度实例（`temperature=0`）+ `with_structured_output(Route, method="function_calling")`
- `Route` 用 `Literal` 枚举硬约束去向：`flights_agent / hotels_agent / cars_agent / trips_agent / FINISH`
- 节点只写 `next`，**不往 messages 写任何 tool_calls**（关键纪律，见 4.4）
- FINISH 时若模型给了 `final_reply` 且本轮尚无专家答复，才追加一条收尾 AIMessage（打招呼/闲聊场景）

### 3.3 子图工厂（graph_chat/agents/base.py）

```python
create_domain_agent(domain_prompt, safe_tools, sensitive_tools)
```

四域共用一套手写结构：`llm` 节点（域 prompt + UserContext + 当前时间）→ 条件路由 → 双 ToolNode → 回 `llm`。新增一个域 = 新建一个十几行的文件（prompt + 工具清单）。

### 3.4 工具归属

| 域 | safe（免审批） | sensitive（中断审批） |
|----|----------------|----------------------|
| 航班 | fetch_user_flight_information、search_flights、lookup_policy | update_ticket_to_new_flight、cancel_ticket |
| 酒店 | search_hotels | book_hotel、update_hotel、cancel_hotel |
| 租车 | search_car_rentals | book_car_rental、update_car_rental、cancel_car_rental |
| 景点 | search_trip_recommendations | book_excursion、update_excursion、cancel_excursion |

`UserContext(passenger_id)` 经 `context_schema` 贯穿主图与所有子图，工具内的越权校验（机票持有人核验）依赖它。

---

## 4. 路由机制详解（核心）

路由不是一处代码，而是**两级路由体系**：主图负责"去哪个域"，子图负责"走哪个工具节点"。

### 4.1 第一级：supervisor 的 LLM 结构化路由

**实现载体**：`StateGraph.add_conditional_edges` —— 条件边本身只是个分发器，真正的决策在节点里：

```python
builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],      # 条件边只读 supervisor 写好的决策
    AGENT_NODES + [END],
)
```

**决策产生过程**（supervisor_node 内部）：

1. 组装输入：`[SystemMessage(主管prompt)] + state["messages"]`（全量历史）
2. `supervisor_llm.invoke(...)` —— 一个普通的 LLM 调用，唯一特殊点是输出被**结构化约束**：
   - `with_structured_output(method="function_calling")` 底层把 `Route` TypedDict 转成一个"工具 schema"喂给模型，模型以 function call 形式返回参数，LangChain 解析成 dict
   - `next_agent` 字段被 `Literal` 五选一约束，**模型不可能路由到不存在的节点**（约束了输出空间 = 路由表的合法性由类型系统保证）
3. 写回状态：`{"next": "__end__" or "xxx_agent"}`，条件边随即完成分发

**决策的三种情形**（这就是"路由"的完整含义，不只是分发）：

| 情形 | 触发时机 | 决策 |
|------|---------|------|
| 意图分发 | 用户输入后第一次 | 读历史识别诉求 → 派给对应域 |
| 复命再决策 | 每次专家回边后 | 看专家产出：任务完成且有新诉求 → 继续派下一个域；否则 FINISH |
| 终止判断 | 每次复命后 | 所有诉求处理完 → 写 `__end__`，本轮结束 |

所以 supervisor 本质是**逐轮决策的 LLM 路由器**：路由的对象不只是"用户意图"，还包括"专家干完活之后的下一步"。跨域任务（改签+订酒店）能工作，靠的是情形 2 的循环——子图 → supervisor → 子图 → ... 直到 FINISH。

**为什么路由决策是"逐轮"的而不是一次性规划**：用户诉求会随对话演化（"改签完算了不签了"），一次性规划无法应对；逐轮决策让每一步都基于最新事实，代价是每跳一次 = 一次 LLM 调用。

### 4.2 路由的确定性保障（四个防线）

| 防线 | 机制 | 防什么 |
|------|------|--------|
| 输出约束 | `Literal` 枚举 + function calling | 模型输出非法节点名 / 幻觉出 sixth agent |
| 温度 | supervisor 独立实例 `temperature=0` | 同样输入路由摇摆 |
| 硬保险 | 主图 `recursion_limit=50` | supervisor 空转死循环（两个域互踢皮球） |
| 确定性保险丝 | supervisor_node 内判断"最后一条已是专家 AI 答复 → 丢弃 final_reply" | 模型违反 prompt 规则、在专家答复后画蛇添足 |

注意最后一道：prompt 规则（6 条决策规则）只是软约束，**凡是有成本的违规都用代码兜底**，这是手写模式相对封装库最大的可控性优势。

### 4.3 第二级：子图内的工具分流路由

```python
def route_conditional_tools(state: State):
    if tools_condition(state) == END:        # llm 没产生 tool_calls → 对话答复 → 子图结束
        return END
    if any(tc["name"] in sensitive_names     # 任一 tool_call 是写操作 → 整批走敏感节点
           for tc in state["messages"][-1].tool_calls):
        return "sensitive_tools"
    return "safe_tools"
```

- `tools_condition` 是 LangGraph 预置的"有没有 tool_calls"判断
- 敏感判断用 `any()` 而非 `tool_calls[0]`——修复了旧版混合调用（一条消息同时调安全+敏感工具）只看第一个的误路由问题
- 路由到 `sensitive_tools` 节点后并不立即执行：编译期 `interrupt_before=["sensitive_tools"]` 会让图在该节点前暂停，等待主循环人工审批（批准→执行；拒绝→注入 ToolMessage 后继续）

### 4.4 一个重要的"负设计"：为什么不用 handoff 工具做路由

社区常见的 supervisor 实现（langgraph-supervisor 库、swarm 模式）是把"转接"做成 supervisor 的**工具**（`transfer_to_flights(...)`），靠 tool_calls 路由。本实现刻意不用，原因：

1. supervisor 产生的 `AIMessage(tool_calls)` 会进入共享 messages 历史，而它永远不会得到对应的 ToolMessage（转接不是真的函数调用）
2. 历史里残留"有 tool_calls 但无应答"的消息后，任何后续 LLM 调用都会触发 `400 insufficient tool messages`——本项目在单 Agent 时代就踩过这个坑
3. 结构化输出返回的是解析结果，不污染历史，路由决策只体现在 `next` 字段上

> 若坚持 handoff 工具方案，必须在每次子图返回后为主管的 tool_call 补一条 ToolMessage（`as_node="supervisor"`），复杂度更高。

### 4.5 路由的成本模型

- 一轮单域对话：supervisor 调用 = 跳数 + 1 次（至少 2 次：分发 + FINISH），专家内部另有若干次
- 一轮 N 跳跨域对话：supervisor 调用 = N + 1 次
- 降级预案（未实现）：简单意图可先用关键词规则直连子 Agent，LLM 只兜底模糊请求

---

## 5. 中断审批（跨子图的横切机制）

```
子图 llm 产生敏感 tool_call
  → 子图在 sensitive_tools 节点前暂停(interrupt_before)
  → 中断冒泡: 主图在该子图节点处暂停
  → 主循环 handle_interrupt():
      root = graph.get_state(config, subgraphs=True)   # 关键: 拿两级状态
      从 root.tasks[].state.next 含 "sensitive_tools" 的子图状态里
      取 messages[-1].tool_calls → 打印工具名+参数 → y/n 审批
      ├─ y: graph.invoke(None, ...)  从主图断点续跑, 子图自动恢复
      └─ n: graph.update_state(子图namespaced config,
                               ToolMessage("用户拒绝了该操作"),
                               as_node="sensitive_tools") → 续跑
```

拒绝路径必须补 ToolMessage，否则历史里残留无应答的 tool_calls，下一轮 400。

## 6. 一次跨域对话的完整时序

以「查一下我的航班信息，然后帮我预订ID为1的酒店」为例（实测通过）：

1. 用户输入入列 messages
2. supervisor 决策1 → `next=flights_agent`
3. 航班子图：llm → fetch_user_flight_information → llm → 输出航班表格，声明"订酒店不归我管" → 子图 END
4. 回边复命 → supervisor 决策2 → `next=hotels_agent`
5. 酒店子图：llm → search → llm → tool_call(book_hotel) → **中断**
6. 主循环打印待审批信息 → 用户批准 → 续跑 → 预订落库 → llm 输出确认
7. 回边复命 → supervisor 决策3 → FINISH → END

## 7. 已知问题与演进方向

| 类别 | 问题 | 状态 |
|------|------|------|
| 数据层 | 改签后 fetch_user_flight_information 返回空（boarding_passes 未同步 flight_id，JOIN 断裂）；改签会把往返两段都改成同一航班 | 待修 |
| 模型层 | deepseek-v4.1-flash 输出 `<think>` 思考文本与特殊标记（`</invoke>` 等），打印层已做正则过滤但不彻底 | 建议端点侧关闭思考 |
| 路由层 | 跨域场景偶发不遵守"继续转接"规则，提前 FINISH | prompt 调优 |
| 架构 | messages 全量共享导致 token 膨胀 | 二阶段：supervisor 传摘要 + 子图私有消息通道 |
| 架构 | 每跳一次 supervisor LLM 调用的固定成本 | 可加规则直连降级 |

演进方向：当前为 BASELINE Subagent；若未来出现"打包行程"类需求（专家需互相感知进度、共享任务清单），再向 Agent Teams 演进，届时新增"共享任务清单字段 + 专家间转交边"，现有结构可平滑扩展。
