import os
from langchain_chroma import Chroma
from ..utils.config_handler import chroma_config
from ..model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from ..utils.logger_handler import logger
from ..utils.path_tool import get_abs_path
from langchain_core.documents import Document


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_config["persist_directory"]),
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_kwargs={"k": chroma_config["k"]},
        )

    def load_document(self):
        # 计算md5 存储
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_config["md5_hex_store"])):
                open(get_abs_path(chroma_config["md5_hex_store"]), "w").close()
                return False

            with open(get_abs_path(chroma_config["md5_hex_store"])) as f:
                for line in f:
                    if line.strip() == md5_for_check:
                        return True
                return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_config["md5_hex_store"]), "a") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith(".txt"):
                return txt_loader(read_path)
            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str]= listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config["allow_knowledge_file_type"]),
            )

        print(allowed_files_path)

        for path in allowed_files_path:
            # 获取文件md5
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"加载知识库 文件 {path} 已存在，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"加载知识库 都没读到任何东西 文件 {path} 为空，跳过")
                    continue
                # 切分文档
                split_document: list[Document] = self.spliter.split_documents(documents)
                # 加载文档到向量数据库
                self.vector_store.add_documents(split_document)
                # 保存md5 避免重复加载
                save_md5_hex(md5_hex)
                logger.info(f"加载知识库 文件 {path} 成功")

            except Exception as e:
                logger.error(f"加载知识库 文件 {path} 失败 {e}", exc_info=True)
                continue


if __name__ == "__main__":
    vector_store_service = VectorStoreService()
    vector_store_service.load_document()

    retriever = vector_store_service.get_retriever()
    res = retriever.invoke("我迷路了")

    for r in res:
        print(r.page_content)
        print("="*20)