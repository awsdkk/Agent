"""
基于streamlit完成的web网页上传服务
用于开发网页
 pip install streamlit
"""

import streamlit as st
from knowledge_base import KnowledgeBaseService

# 添加网页标题
st.title("知识库更新 文件上传服务")

# 文件上传框
uploaded_file = st.file_uploader(
    "请上传txt文件",
    type="txt",
    accept_multiple_files=False,    # 是否接受多个文件上传
)


# session_state 用于存储网页会话中的变量 是一个字典
if "knowledgebaseservice" not in st.session_state:
    st.session_state["knowledgebaseservice"] = KnowledgeBaseService()

if uploaded_file is not None:
    # 读取上传的文件内容
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    file_size = uploaded_file.size / 1024  # 转换为KB
    file_content = uploaded_file.read().decode("utf-8-sig")

    # 显示文件信息
    st.write(f"文件大小: {file_size:.2f} KB")
    st.write(f"文件类型: {file_type}")
    st.subheader(f"文件名: {file_name}")
    st.write("文件内容预览:")
    st.write(file_content[:200])  # 显示200个字符内容

    text = uploaded_file.getvalue().decode("utf-8")

    # 调用知识库服务上传文件
    res = st.session_state["knowledgebaseservice"].upload_by_str(text, file_name)
    st.write(res)
# 当web页面发生变化 streamlit代码会重头重新跑一遍 要避免这种情况 用session_state存储变量

