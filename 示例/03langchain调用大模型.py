from langchain_classic.chains.question_answering.map_reduce_prompt import messages
from langchain_community.chat_models import ChatTongyi
from langchain_community.llms.tongyi import Tongyi
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

print("=" * 60)
print("🤖 通义千问模型测试")
print("=" * 60)

# 第一部分：普通调用
print("\n📄 普通调用测试：")
print("问题：简单解释一下你自己")
print("-" * 40)

model = Tongyi(model="qwen-max")
res = model.invoke(input="简单解释一下你自己")
print(f"回答：{res}")

# 第二部分：流式输出
print("\n" + "=" * 60)
print("🌊 流式输出测试：")
print("问题：复杂的解释一下你自己")
print("-" * 40)

res = model.stream(input="复杂的解释一下你自己")
print("回答：", end="", flush=True)
for chunk in res:
    print(chunk, end="", flush=True)
print()  # 换行

# 第三部分：Chat模型测试
print("\n" + "=" * 60)
print("💬 Chat模型测试：")
print("-" * 40)

chat_model = ChatTongyi(model="qwen-max")

# 使用完整的消息类形式，更加清晰
messages = [
    SystemMessage(content="你是一个专业的诗人，擅长创作优美的诗歌"),
    HumanMessage(content="给我做出一些歌颂三体世界的诗歌")
]

print("系统设定：你是一个专业的诗人，擅长创作优美的诗歌")
print("用户提问：给我做出一些歌颂三体世界的诗歌")
print("-" * 40)
print("AI回答：")

res = chat_model.stream(messages)
for chunk in res:
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()  # 最终换行

print("\n" + "=" * 60)
print("✅ 测试完成！")