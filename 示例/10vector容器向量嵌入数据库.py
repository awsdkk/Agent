"""
VectorStores向量存储测试示例 - 使用Chroma
此脚本演示了如何使用LangChain与Chroma向量数据库进行文本嵌入和相似性搜索
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.docstore.document import Document
import os

def test_chroma_vector_store():
    """
    测试Chroma向量存储的基本功能
    """
    print("开始测试Chroma向量存储功能...")

    # 示例文档数据
    documents = [
        "人工智能是计算机科学的一个分支，致力于让机器具备智能行为。",
        "机器学习是人工智能的一个子领域，专注于算法和统计模型。",
        "深度学习是机器学习的一种方法，基于人工神经网络。",
        "自然语言处理是人工智能的一个应用领域，涉及人机语言交互。",
        "计算机视觉是让计算机理解和解释视觉信息的技术。",
        "强化学习是一种机器学习范式，通过奖励机制训练智能体。",
        "卷积神经网络在图像识别任务中表现优异。",
        "循环神经网络擅长处理序列数据如时间序列或文本。"
    ]

    # 创建文档对象
    docs = [Document(page_content=text, metadata={"source": f"doc_{i}"}) for i, text in enumerate(documents)]

    print(f"已准备{len(docs)}个文档用于向量化")

    try:
        # 初始化嵌入模型
        embeddings = OpenAIEmbeddings()

        # 创建Chroma向量存储
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./chroma_db"  # 可选：指定持久化目录
        )

        print("Chroma向量数据库创建成功！")

        # 执行相似性搜索
        query = "什么是机器学习？"
        similar_docs = vectorstore.similarity_search(query, k=3)

        print(f"\n查询: {query}")
        print("最相似的文档:")
        for i, doc in enumerate(similar_docs):
            print(f"{i+1}. {doc.page_content} (来源: {doc.metadata.get('source', 'unknown')})")

        # 演示带分数的相似性搜索
        similar_docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=3)

        print("\n带相似度分数的搜索结果:")
        for i, (doc, score) in enumerate(similar_docs_with_scores):
            print(f"{i+1}. {doc.page_content} (相似度分数: {score:.4f})")

        # 演示最大边际相关性搜索(MMR)
        mmr_results = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=5)

        print("\nMMR搜索结果 (多样性更好的结果):")
        for i, doc in enumerate(mmr_results):
            print(f"{i+1}. {doc.page_content}")

        # 演示混合搜索 (如果支持)
        print(f"\n向量数据库中的总文档数: {vectorstore._collection.count()}")

        return vectorstore

    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        print("可能原因: 未设置API密钥、缺少相关依赖包或Chroma配置问题")
        return None

def test_chroma_with_custom_collection():
    """
    测试带有自定义集合名称的Chroma向量存储
    """
    print("\n" + "="*50)
    print("测试自定义集合名称的Chroma向量存储")
    print("="*50)

    # 准备不同的文档集合
    tech_docs = [
        "Python是一种高级编程语言，以其简洁的语法而闻名。",
        "JavaScript是Web开发的核心技术之一。",
        "机器学习算法可以从数据中自动学习模式。",
        "云计算提供了按需访问计算资源的能力。"
    ]

    science_docs = [
        "DNA是携带遗传信息的分子。",
        "量子力学描述了微观粒子的行为。",
        "进化论解释了物种的起源和发展。",
        "相对论改变了我们对时空的理解。"
    ]

    from langchain_community.docstore.document import Document

    # 创建科技类文档
    tech_documents = [Document(page_content=text, metadata={"category": "technology", "source": f"tech_doc_{i}"})
                     for i, text in enumerate(tech_docs)]

    # 创建科学类文档
    science_documents = [Document(page_content=text, metadata={"category": "science", "source": f"science_doc_{i}"})
                        for i, text in enumerate(science_docs)]

    try:
        # 初始化嵌入模型
        embeddings = OpenAIEmbeddings()

        # 创建第一个Chroma实例（科技类文档）
        tech_vectorstore = Chroma.from_documents(
            documents=tech_documents,
            embedding=embeddings,
            collection_name="tech_documents",
            persist_directory="./chroma_tech_db"
        )

        # 创建第二个Chroma实例（科学类文档）
        science_vectorstore = Chroma.from_documents(
            documents=science_documents,
            embedding=embeddings,
            collection_name="science_documents",
            persist_directory="./chroma_science_db"
        )

        print("两个不同类别的Chroma集合创建成功！")

        # 测试查询
        query = "编程语言"
        tech_results = tech_vectorstore.similarity_search(query, k=2)
        science_results = science_vectorstore.similarity_search(query, k=2)

        print(f"\n在科技集合中查询 '{query}':")
        for i, doc in enumerate(tech_results):
            print(f"  {i+1}. {doc.page_content}")

        print(f"\n在科学集合中查询 '{query}':")
        for i, doc in enumerate(science_results):
            print(f"  {i+1}. {doc.page_content}")

        return tech_vectorstore, science_vectorstore

    except Exception as e:
        print(f"自定义集合测试失败: {str(e)}")
        return None, None

def test_add_documents_to_chroma():
    """
    测试向现有Chroma数据库添加新文档
    """
    print("\n" + "="*50)
    print("测试向现有Chroma数据库添加文档")
    print("="*50)

    try:
        # 初始化嵌入模型
        embeddings = OpenAIEmbeddings()

        # 首先创建一个基础的Chroma数据库
        initial_docs = [
            Document(page_content="人工智能正在改变世界。", metadata={"source": "initial_1"}),
            Document(page_content="机器学习是AI的重要组成部分。", metadata={"source": "initial_2"})
        ]

        vectorstore = Chroma.from_documents(
            documents=initial_docs,
            embedding=embeddings,
            collection_name="dynamic_collection",
            persist_directory="./chroma_dynamic_db"
        )

        print(f"初始文档数量: {vectorstore._collection.count()}")

        # 添加新的文档
        new_docs = [
            Document(page_content="深度学习在网络架构方面取得了突破。", metadata={"source": "added_1"}),
            Document(page_content="自然语言处理技术不断进步。", metadata={"source": "added_2"})
        ]

        ids = vectorstore.add_documents(new_docs)
        print(f"新增文档后总数: {vectorstore._collection.count()}")
        print(f"新增的文档ID: {ids}")

        # 验证添加的文档
        all_docs = vectorstore.similarity_search("", k=10)  # 获取所有文档
        print(f"\n数据库中的所有文档:")
        for i, doc in enumerate(all_docs):
            print(f"{i+1}. {doc.page_content} (来源: {doc.metadata.get('source')})")

        return vectorstore

    except Exception as e:
        print(f"添加文档测试失败: {str(e)}")
        return None

def advanced_query_examples(vectorstore):
    """
    演示高级查询功能
    """
    if vectorstore is None:
        print("无法执行高级查询测试，向量存储为空")
        return

    print("\n" + "="*50)
    print("高级查询功能演示")
    print("="*50)

    try:
        # 基于过滤条件的查询
        filtered_results = vectorstore.similarity_search(
            "学习",
            k=3,
            filter={"source": {"$regex": "doc_"}}
        )

        print("带过滤条件的查询结果:")
        for i, doc in enumerate(filtered_results):
            print(f"{i+1}. {doc.page_content} (来源: {doc.metadata.get('source')})")

        # MMR搜索参数调整
        mmr_results = vectorstore.max_marginal_relevance_search(
            "人工智能",
            k=3,
            fetch_k=10,
            lambda_mult=0.5  # 多样性和相关性的平衡参数
        )

        print(f"\nMMR搜索结果 (lambda_mult=0.5):")
        for i, doc in enumerate(mmr_results):
            print(f"{i+1}. {doc.page_content}")

    except Exception as e:
        print(f"高级查询测试失败: {str(e)}")

if __name__ == "__main__":
    print("=" * 60)
    print("Chroma向量存储功能测试")
    print("=" * 60)

    # 测试基本功能
    chroma_store = test_chroma_vector_store()

    # 测试自定义集合
    tech_store, science_store = test_chroma_with_custom_collection()

    # 测试动态添加文档
    dynamic_store = test_add_documents_to_chroma()

    # 演示高级功能
    advanced_query_examples(chroma_store)

    print("\n所有测试完成！")

    # 提供清理说明
    print("\n清理说明:")
    print("- Chroma数据库文件保存在当前目录下的相应文件夹中")
    print("- 如需删除数据库，可手动删除 chroma_db, chroma_tech_db, chroma_science_db, chroma_dynamic_db 文件夹")
