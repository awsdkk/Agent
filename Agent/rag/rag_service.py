'''总结服务类
用户提问 搜索参考资料 将提问和参考资料提交给模型 让模型回复
'''
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from .vector_store import VectorStoreService
from ..utils.prompt_loader import load_rag_prompt
from ..model.factory import chat_model

def print_prompt(prompt):
    print(prompt.to_string())
    print("=" * 20)
    return prompt

class RagSummarizeService(object):
    def __init__(self):
        self.vector_store_service = VectorStoreService()
        self.retriever = self.vector_store_service.get_retriever()
        self.prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(
            self.prompt_text,
        )
        self.model = chat_model
        self.chain = self._init_chain()

    # 初始化链
    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain
    # 检索文档获取
    def retrieve_docs(self, query: str) -> list[Document]:
        chat = self.retriever.invoke(query)
        return chat

    # 调用链 总结
    def rag_summarize(self, query: str) -> str:
        context_docs = self.retrieve_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"[[参考资料{counter}]] 参考内容: {doc.page_content} 参考元数据: {doc.metadata} \n"

        return self.chain.invoke({
            "input": query,
            "context": context,
        })



if __name__ == "__main__":
    service = RagSummarizeService()
    res = service.rag_summarize("小户型适合哪些扫地机器人")
    print(res)