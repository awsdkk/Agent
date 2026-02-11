import os
from langchain_core.tools import tool
from ...rag.rag_service import RagSummarizeService
import random
from ...utils.config_handler import agent_config
from ...utils.logger_handler import logger
from ...utils.path_tool import get_abs_path

ragService = RagSummarizeService()
user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008"]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]
external_data = {}

@tool(description="从向量库中检索参考资料")
def rag_summarize(query: str) -> str:
    return ragService.rag_summarize(query)

@tool(description="获取指定城市的天气 以字符串形式返回")
def get_weather(city: str) -> str:
    return f"城市{city}的天气是晴朗 温度26° 空气湿度49% 南风一级 AQI21"


@tool(description="获取用户定位 以字符串形式返回")
def get_user_location() -> str:
    return random.choice(["北京", "上海", "广州", "深圳"])

@tool(description="获取用户ID 以字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)

@tool(description="获取当前月份 以字符串形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)

# csv文件处理一下
def generate_external_data():
    if not external_data:
        external_data_path = get_abs_path(agent_config["external_data_path"])
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件 {external_data_path} 不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")
                user_id: str = arr[0].replace('"',"")
                feature: str = arr[1].replace('"',"")
                efficiency: str = arr[2].replace('"',"")
                consumables: str = arr[3].replace('"',"")
                comparison: str = arr[4].replace('"',"")
                time: str = arr[5].replace('"',"")

                if user_id not in external_data:
                    external_data[user_id] = {}
                # 不能在if里
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                    }

@tool(description="获取指定用户的指定月份报告 以字符串形式返回 如果没有报告 返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"没能检索到用户使用记录 用户 {user_id} 在月份 {month} 没有报告")
        return ""

@tool(description="无入参 无返回值 调用后触发中间件自动为报告生成的场景注入上下文信息 为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已经调用"