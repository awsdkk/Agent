from langchain_classic.chains.summarize.map_reduce_prompt import prompt_template
from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

print("🍳 厨师创作助手")
print("=" * 50)

# 初始化模型
model = Tongyi(model="qwen-max")

# 第一部分：基础模板使用
print("\n📋 基础模板测试：")
print("输入：鱼香肉丝、肉丝、鱼香")
print("-" * 30)

chef_template = PromptTemplate.from_template(
    "你是一个专业厨师，你需要根据用户的输入创作一道菜。\n"
    "用户的输入是：{a}、{b}、{c}\n"
    "请详细描述这道菜的制作方法和特色。"
)

prompt_text = chef_template.format(a="鱼香肉丝", b="肉丝", c="鱼香")
result = model.invoke(prompt_text)
print(result)

print("\n" + "=" * 50)

# 第二部分：链式操作
print("\n🔗 链式操作测试：")
print("输入：椅子、初音未来、土块")
print("-" * 30)

# 创建链：模板 -> 模型
chef_chain = chef_template | model

# 使用链式调用
result = chef_chain.invoke({"a": "椅子", "b": "初音未来", "c": "土块"})
print(result)

print("\n✅ 测试完成！")