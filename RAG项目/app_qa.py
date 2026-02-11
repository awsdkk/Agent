import time
from rag import RagService
import streamlit as st
from langchain_classic.chains.summarize.map_reduce_prompt import prompt_template
import config_data as config

# 标题
st.title("智能客服")
st.divider()    # 分隔线

if "rag_service" not in st.session_state:
    st.session_state["rag_service"] = RagService()


if "message" not in st.session_state:
    st.session_state["message"] = [
        {
            "role": "assistant",
         "content": "你好，我是智能客服，有什么我可以帮助你的吗？"
        }
    ]

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])


prompt = st.chat_input()

if prompt:
    # 在页面输出用户提问
    st.chat_message("user").write(prompt)
    # 加入用户提问到会话状态
    st.session_state["message"].append(
        {
        "role": "user",
        "content": prompt
        }
    )

    ai_res_list = []
    with st.spinner("正在思考中..."):
        res_stream = st.session_state["rag_service"].chain.invoke(
            {"input": prompt},
            config=config.session_config
        )

        # 因为res_stream是一个生成器，所以需要将其转换为字符串
        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message("assistant").write_stream(capture(res_stream, ai_res_list))
        st.session_state["message"].append({"role": "assistant","content": "".join(ai_res_list)})

