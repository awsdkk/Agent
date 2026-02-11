from langchain_community.chat_models import ChatTongyi
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

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

store = {}

# 定义一个函数 通过session_id获取历史会话
def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

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

    res = conversation_chain.invoke(
        {"input": "网速好真的很不错"},
        config = session_config,
    )
    print(res)

    res = conversation_chain.invoke(
        {"input": "我也很喜欢"},
        config = session_config,
    )
    print(res)

    res = conversation_chain.invoke(
        {"input": "你截止到现在的回答进行了几轮对话"},
        config = session_config,
    )
    print(res)













