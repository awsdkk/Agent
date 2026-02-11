from langchain.agents import create_agent
from ..model.factory import chat_model
# from Agent.model.factory import chat_model
from .tools.agent_tools import *
from .tools.middleware import *


class react_agent():
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id, get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_swtich],
        )

    # 流式执行
    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
        # 第三个参数context 就是上下文中的runtime中的信息 我们要切换提示词的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            yield latest_message.content.strip() + "\n"

if __name__ == "__main__":
    agent = react_agent()
    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
