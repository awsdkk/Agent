from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoresService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

def print_prompt(prompt_template):
    print("="*20)
    print(prompt_template)
    print("="*20)
    return prompt_template


class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoresService(
            embedding=DashScopeEmbeddings(model = config.embedding_model_name),
        )

        # 提示词模板 注入 上下文 历史记录 用户输入
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以已知的参考资料为主 简洁干净的回答问题 参考资料是{context}"),
                ("system", "并且我提供用户对话历史记录 {history}"),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问 {input}"),
            ]
        )

        self.chat_model = ChatTongyi(model=config.chat_model_name)

        self.chain = self.__get_chain()



    def __get_chain(self):
        '''获取链'''
        retriever = self.vector_service.get_retriever()

        # 组装成字符串 检索器 -> 格式化字符串 -> 字符串 如果没有不拼接
        def format_document(docs: list[Document]):
            if not docs:
                return "无有相关资料"
            formatted_str = ""
            for doc in docs:
                formatted_str += f"参考资料{doc.metadata['source']} 内容{doc.page_content}\n"

            return formatted_str

        def get_query_string(x):
            """
            从各种可能的输入格式中提取出查询字符串
            """
            value = x.get("input")

            # 最常见的情况：RunnableWithMessageHistory 包装成了 list[BaseMessage]
            if isinstance(value, list) and value and hasattr(value[0], "content"):
                return value[0].content

            # 直接是 HumanMessage
            if hasattr(value, "content"):
                return value.content

            # 已经是字符串了（测试或非对话场景）
            if isinstance(value, str):
                return value

            # 兜底，转字符串（通常不会走到这里）
            return str(value)

        def format_for_prompt_template(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        # 链RunnableLambda(format_for_retriever)
        chain = (
                {
                    "input": RunnablePassthrough(),
                    "context": RunnableLambda(get_query_string) | retriever | format_document
                }
                | RunnableLambda(lambda x: {
            "input": get_query_string(x),  # 保证 prompt 里的 {input} 也是纯字符串
            "context": x["context"],
            "history": x.get("history", []),  # 更安全
        })
                | self.prompt_template
                | print_prompt
                | self.chat_model
                | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
        )


        return conversation_chain

if __name__ == "__main__":
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }
    res = service = RagService().chain.invoke({"input": "你是谁"}, session_config)
    res = service = RagService().chain.invoke({"input": "我们进行几轮对话了"}, session_config)


