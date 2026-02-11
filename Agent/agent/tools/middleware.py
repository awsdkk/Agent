from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from langgraph.runtime import Runtime
from typing import Callable
from ...utils.logger_handler import logger
from ...utils.prompt_loader import load_report_prompt, load_system_prompt


# 工具执行监控
@wrap_tool_call
def monitor_tool(
        # 工具调用请求数据封装 入参
        request: ToolCallRequest,
        # 工具调用处理函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    # 日志的打印
    logger.info(f"工具调用请求 执行工具: {request.tool_call['name']}")
    logger.info(f"工具调用请求 传入参数: {request.tool_call['args']}")

    try:
        res = handler(request)
        logger.info(f"工具调用成功")

        if request.tool_call['name'] == "fill_context_for_report":
            # 为报告生成场景 注入上下文信息
            request.runtime.context["report"] = True

        return res
    except Exception as e:
        logger.error(f"工具调用请求 执行工具 {request.tool_call['name']} 发生错误: {e}")
        raise e

# 在模型调度前输出日志 仅仅打印日志
@before_model
def log_before_model(
        state: AgentState,  # agent状态
        runtime: Runtime,  # 运行时上下文信息
):
    logger.info(f"模型调度前 状态: 有{len(state['messages'])}条消息传递给模型")
    logger.debug(f"模型调度前 状态:消息类型{type(state['messages'][-1]).__name__} 最近一条消息 {state['messages'][-1].content.strip()}")
    logger.info(f"模型调度前 运行时上下文: {runtime}")

    return None

# 每一次在生成提示词之前 调用这个函数
@dynamic_prompt
def report_prompt_swtich(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompt()

    return load_system_prompt()