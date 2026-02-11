import streamlit as st
from agent.react_agent import react_agent

'''
这个项目包装成agent智能客服助手
'''

def capture(generator, cache_list):
    for chunk in generator:
        cache_list.append(chunk)
        yield chunk

# 标题
st.title("智能体助手")
st.divider()

if "agent" in st.session_state:
    agent = st.session_state["agent"] = react_agent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []
# 显示历史消息 为了不重复
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({
        "role": "user",
        "content": prompt
    })

    response_messages = []

    with st.spinner("智能体思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt)

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))

        st.session_state["messages"].append({
            "role": "assistant",
            "content": response_messages[-1]
        })