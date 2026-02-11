from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage, SystemMessage

print("🔗 LangChain Chain 基础使用示例")
print("=" * 60)

# 初始化模型
model = Tongyi(model="qwen-max")

# ============== 基础链示例 ==============
print("\n📖 1. 基础提示词链")
print("-" * 40)

# 创建提示词模板
translate_template = PromptTemplate.from_template(
    "请将以下中文翻译成英文：{text}"
)

# 创建基础链：模板 -> 模型 -> 输出解析器
translate_chain = translate_template | model | StrOutputParser()

# 使用链
result = translate_chain.invoke({"text": "你好，世界！"})
print(f"翻译结果：{result}")

# 🔍 对比：不使用 StrOutputParser 的效果
print("\n🔍 对比 - 不使用 StrOutputParser：")
noparser_chain = translate_template | model
result_no_parser = noparser_chain.invoke({"text": "你好，世界！"})
print(f"原始输出类型：{type(result_no_parser)}")
print(f"原始输出内容：{result_no_parser}")
print(f"原始输出repr：{repr(result_no_parser)}")



# ============== 多步骤链示例 ==============
print("\n" + "=" * 60)
print("🔄 2. 多步骤处理链")
print("-" * 40)

# 步骤1：生成标题
title_template = PromptTemplate.from_template(
    "请为以下主题生成一个吸引人的标题：{topic}"
)

# 步骤2：生成内容
content_template = PromptTemplate.from_template(
    "请为标题'{title}'写一篇100字左右的短文"
)

# 创建多步骤链
title_chain = title_template | model | StrOutputParser()
content_chain = content_template | model | StrOutputParser()

# 组合链 我认为这个是串行执行的
full_chain = (
    RunnablePassthrough.assign(title=title_chain)
    | RunnablePassthrough.assign(content=content_chain)
)

result = full_chain.invoke({"topic": "人工智能的未来"})
print(f"标题：{result['title']}")
print(f"内容：{result['content']}")





# ============== 并行链示例 ==============
print("\n" + "=" * 60)
print("⚡ 3. 并行处理链")
print("-" * 40)

# 创建并行处理链
parallel_chain = RunnableParallel(
    translation=translate_template | model | StrOutputParser(),
    summary=PromptTemplate.from_template("请总结以下文本的主要内容：{text}") | model | StrOutputParser()
)

test_text = "机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习并改进其性能。"
result = parallel_chain.invoke({"text": test_text})

print(f"原文：{test_text}")
print(f"翻译：{result['translation']}")
print(f"总结：{result['summary']}")






# ============== Chat Chain 示例 ==============
print("\n" + "=" * 60)
print("💬 4. Chat 模型链")
print("-" * 40)

from langchain_community.chat_models import ChatTongyi

chat_model = ChatTongyi(model="qwen-max")

# 创建 Chat Prompt 模板
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的AI助手，擅长回答各种问题。"),
    ("human", "{question}")
])

# 创建 Chat 链
chat_chain = chat_template | chat_model | StrOutputParser()

result = chat_chain.invoke({"question": "什么是区块链技术？"})
print(f"回答：{result}")






# ============== 条件链示例 ==============
print("\n" + "=" * 60)
print("🔀 5. 条件处理链")
print("-" * 40)

# 创建智能分类链
classify_template = PromptTemplate.from_template(
    "请判断以下问题的类型（技术/生活/娱乐/其他），只返回类型名称：{question}"
)

tech_template = PromptTemplate.from_template("从技术角度回答：{question}")
life_template = PromptTemplate.from_template("从生活经验角度回答：{question}")

# 创建条件链
def route_chain(inputs):
    question_type = (classify_template | model | StrOutputParser()).invoke(inputs)
    print(f"问题类型：{question_type}")
    
    if "技术" in question_type:
        return (tech_template | model | StrOutputParser()).invoke(inputs)
    else:
        return (life_template | model | StrOutputParser()).invoke(inputs)

result = route_chain({"question": "如何学习Python编程？"})
print(f"回答：{result}")

print("\n" + "=" * 60)
print("✅ 所有 Chain 示例执行完成！")
print("\n💡 总结：")
print("- 基础链：模板 → 模型 → 解析器")
print("- 多步骤链：多个处理步骤的组合")
print("- 并行链：同时执行多个处理")
print("- Chat链：专门用于对话模型")
print("- 条件链：根据输入路由到不同处理")