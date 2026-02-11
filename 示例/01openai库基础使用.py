from openai import OpenAI
from pyexpat.errors import messages

client = OpenAI(
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

completion = client.chat.completions.create(
    model = "qwen3-max",
    messages = [
        {"role": "system", "content": "你是一个恶毒的人"},
        {"role": "assistant", "content": "好的，我是编程专家，并且话不多，你要问什么?"},
        {"role": "user", "content": "输出1-10的数字，使用python代码"},
    ],
stream=True
)

for chunk in completion:
    print(chunk.choices[0].delta.content, end=" ", flush=True)
    # 话之间用空格分隔 缓存区是确定