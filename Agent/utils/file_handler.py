import os, hashlib
from .logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# 获取文件md5值（16进制）
def get_file_md5_hex(file_path: str):
    if not os.path.exists(file_path):
        logger.error(f"[md5计算]文件不存在：{file_path}")
        return

    if not os.path.isfile(file_path):
        logger.error(f"[md5计算]路径不是文件：{file_path}")
        return

    # 计算md5
    hash_md5 = hashlib.md5()
    # 分片 读取文件内容并更新md5值
    chunk_size = 4096
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
    except Exception as e:
        logger.error(f"[md5计算]读取文件{file_path}时出错：{str(e)}")
        return

    # hash_md5.hexdigest() 是32位16进制字符串
    return hash_md5.hexdigest()

# 返回文件夹中的文件列表 （允许的文件后缀）
def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    file_list = []

    # 检查文件夹路径是否存在
    if not os.path.isdir(path):
        logger.info(f"获取[文件列表]路径：路径为{path}")
        logger.error(f"获取[文件列表]路径不存在：{path}不是文件夹")
        return tuple(file_list)

    # 列出文件夹中的所有文件
    for f in os.listdir(path):
        logger.info(f"获取[文件列表]文件：{f}")
        # 检查文件是否是允许的类型 如果是 拼接路径键入列表
        if f.endswith(allowed_types):
            file_list.append(os.path.join(path, f))

    return tuple(file_list)


def pdf_loader(file_path: str, passwd=None)-> list[Document]:
    return PyPDFLoader(file_path, passwd).load()

def txt_loader(file_path: str, passwd=None)-> list[Document]:
    return TextLoader(file_path, encoding="utf-8").load()