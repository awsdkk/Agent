import os, json
from typing import Sequence

from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_community.chat_models import ChatTongyi
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import message_to_dict, messages_from_dict, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory


# 文件会话记忆 实现长期会话记忆
class FileChatMessage(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id
        self.storage_path = storage_path
        # 初始化完整文件路径
        self.file_path = os.path.join(self.storage_path, f"{self.session_id}.json")
        # 确保存储路径存在 不存在就创建
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    # 实现方法
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # sequence序列 类似list tuple
        all_messages = list(self.messages)
        # 老的加上新的
        all_messages.extend(messages)
        # 保存到文件 先处理数据 变成字典了 方便存储
        new_messages = []
        for msg in all_messages:
            new_messages.append(message_to_dict(msg))

        # 将数据 变成json 写入文件
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f, ensure_ascii=False, indent=2)

    # 通过装饰器把messages方法变成成员属性
    @property
    def messages(self) -> Sequence[BaseMessage]:
        # 从文件读取数据
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messsages_data = json.load(f)
                # 字典转换为消息对象
                return messages_from_dict(messsages_data)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        # 清空文件内容
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


model = ChatTongyi(model="qwen3-max")

prompt_template = PromptTemplate.from_template(
    "你需要根据历史会话来回应用户问题。 历史会话：{chat_history} 用户问题：{input} 请回答"
)

chatprompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个专业的客服，需要根据历史会话来回应用户问题。"),
        MessagesPlaceholder("chat_history"),
        ("human", "回答如下问题{input}"),
    ]

)

str_parser = StrOutputParser()

def print_prompt(full_prompt):
    print("-" * 20)
    print(full_prompt.to_string())
    print("-" * 20)
    return full_prompt


base_chain = chatprompt_template | print_prompt | model | str_parser


# 定义一个函数 通过session_id获取历史会话
def get_history(session_id):
    # 创建一个文件会话记忆对象
    return FileChatMessage(session_id, "./chat_history")

# 创建一个新的链
conversation_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


if __name__ == "__main__":
    # 添加langchain的配置 固定格式
    session_config = {
        "configurable": {
            "session_id": "u123"
        }
    }

    # res = conversation_chain.invoke(
    #     {"input": "网速好真的很不错"},
    #     config = session_config,
    # )
    # print(res)
    #
    # res = conversation_chain.invoke(
    #     {"input": "我也很喜欢"},
    #     config = session_config,
    # )
    # print(res)

    res = conversation_chain.invoke(
        {"input": "那现在的回答进行了几轮对话"},
        config = session_config,
    )
    print(res)

