'''
知识库 knowledge_base.py
'''
import hashlib
import os
import config_data as config
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime


def check_md5(md5_str: str):
    # 检测传入的文件是否处理过了
    # false未处理 ture处理
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path, 'r', encoding='utf-8').readlines():
            line = line.strip() # 处理字符串前后空格和回车
            if line == md5_str:
                return True

        return False

    pass

def save_md5(md5_str: str):
    # 传入md5保存到text
    with open(config.md5_path, 'a', encoding='utf-8') as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str: str, encoding='utf-8'):
    # 字符串转成md5 字符串-> 字节流 -> md5
    str_bytes = input_str.encode(encoding=encoding)

    # 创建md5对象
    md5_obj = hashlib.md5(str_bytes)

    # 返回md5十六进制字符串
    return md5_obj.hexdigest()


class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chroma = Chroma(
            collection_name = config.collection_name, #数据库配置
            embedding_function = DashScopeEmbeddings(model = "text-embedding-v4"),
            persist_directory = config.persist_directory,   #数据库本地存储文件夹
        )
        # 文本分割器对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,   # 当前分割文本段最大长度
            chunk_overlap=config.chunk_overlap,   #允许字符重叠数量
            separators=config.separators,   # 符号划分
            length_function=len,
        )

    def upload_by_str(self, data, filename):
        # 传入数据向量化 存入数据库中
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过] 内容已存在知识库中"

        # 文本分割
        if len(data) > config.max_split_char_number:
            knowledge_chunks = self.spliter.split_text(data)    #列表 包着字符串 类型
        else:
            knowledge_chunks = [data]


        ''' 为什么加上元数据
        1. 支持过滤检索（最重要、最常用的原因）
        向量相似度搜索（semantic search）虽然强大，但很容易召回“语义相关但实际上不该出现的”内容。
        加了 metadata 后，你可以在检索阶段先做硬过滤，再做向量相似度排序，大幅提高召回的精准度。
        '''
        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "用户在此",
        }

        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        # 数据处理完了记录到md5text中去
        save_md5(md5_hex)

        return "[成功] 内容已经成功载入向量库"


if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str("大师傅士大夫", "file01")
