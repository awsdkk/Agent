import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # api_key="sk-6d8efd80c0274c1dbd2c7c05f701de50",
    base_url="http://localhost::11434/v1",
)
completion = client.chat.completions.create(
    model="对应的本地模型名字 这个只是备用 还是用云模型的",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
    stream=True
)
for chunk in completion:
    print(chunk.choices[0].delta.content, end="", flush=True)