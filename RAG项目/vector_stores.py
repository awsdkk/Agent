from openai import models

from knowledge_base import KnowledgeBaseService
import config_data as config
from langchain_chroma import Chroma



class VectorStoresService(object):
    def __init__(self, embedding):
        self.embedding = embedding

        self.vector_stores = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        '''获取检索器'''
        return self.vector_stores.as_retriever(search_kwargs={"k": config.similarity_threshold})


if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    service = VectorStoresService(
        DashScopeEmbeddings(model="text-embedding-v4")
    )
    retriever = service.get_retriever()

    res = retriever.invoke("我的178")
    print(res)