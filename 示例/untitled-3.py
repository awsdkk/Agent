from langchain_community.document_loaders import JSONLoader
import json

# 首先确认我们的测试数据
test_data = {
    "users": [
        {
            "id": 1,
            "name": "张三",
            "age": 25,
            "city": "北京",
            "job": "工程师",
            "email": "zhangsan@example.com",
            "skills": ["Python", "JavaScript", "SQL"]
        },
        {
            "id": 2,
            "name": "李四",
            "age": 30,
            "city": "上海",
            "job": "设计师",
            "email": "lisi@example.com",
            "skills": ["Photoshop", "Illustrator", "UI/UX"]
        }
    ],
    "company": "测试公司",
    "location": "中国",
    "departments": ["技术部", "设计部", "教学部", "医疗部"],
    "total_employees": 4
}

# 保存测试数据
with open('test_data.json', 'w', encoding='utf-8') as json_file:
    json.dump(test_data, json_file, ensure_ascii=False, indent=2)

print("=== LangChain JSONLoader 测试 ===\n")


# 测试1: 基本加载 - 提取用户列表中的每个用户
print("1. 基本加载测试 - 提取users数组中的每个元素:")
loader1 = JSONLoader(
    file_path="test_data.json",
    jq_schema=".users[]",  # 提取users数组中的每个元素
    text_content=False     # 不将内容作为纯文本处理
)
docs1 = loader1.load()
print(f"加载了 {len(docs1)} 个文档")
for i, doc in enumerate(docs1):
    print(f"文档 {i+1}: {doc.page_content}")
    print(f"元数据: {doc.metadata}\n")



# 测试2: 加载整个对象 - 修正版
print("2. 加载整个JSON对象 (使用content_key):")
loader2 = JSONLoader(
    file_path="test_data.json",
    jq_schema=".",  # 加载整个JSON对象
    content_key="company",  # 使用company字段作为内容
    text_content=True
)
docs2 = loader2.load()
print(f"加载了 {len(docs2)} 个文档")
for i, doc in enumerate(docs2):
    print(f"文档 {i+1}: {doc.page_content}")
    print(f"元数据: {doc.metadata}\n")

# 或者不使用text_content（推荐用于复杂对象）
print("2b. 加载整个JSON对象 (不使用text_content):")
loader2b = JSONLoader(
    file_path="test_data.json",
    jq_schema=".",  # 加载整个JSON对象
    text_content=False
)
docs2b = loader2b.load()
print(f"加载了 {len(docs2b)} 个文档")
for i, doc in enumerate(docs2b):
    content_preview = str(doc.page_content)[:100] + "..." if len(str(doc.page_content)) > 100 else str(doc.page_content)
    print(f"文档 {i+1} 内容预览: {content_preview}")
    print(f"元数据: {doc.metadata}\n")





# 测试3: 提取特定字段
print("3. 提取特定字段 (只提取用户姓名和城市):")
loader3 = JSONLoader(
    file_path="test_data.json",
    jq_schema='.users[] | {name: .name, city: .city}',
    text_content=False
)
docs3 = loader3.load()
print(f"加载了 {len(docs3)} 个文档")
for i, doc in enumerate(docs3):
    print(f"文档 {i+1}: {doc.page_content}")
    print(f"元数据: {doc.metadata}\n")





# 测试4: 使用content_key参数
print("4. 使用content_key参数指定内容键:")
loader4 = JSONLoader(
    file_path="test_data.json",
    jq_schema=".users[]",
    content_key="name",  # 只提取name字段作为内容
    text_content=True
)
docs4 = loader4.load()
print(f"加载了 {len(docs4)} 个文档")
for i, doc in enumerate(docs4):
    print(f"文档 {i+1}: {doc.page_content}")
    print(f"元数据: {doc.metadata}\n")




# 测试5: 处理简单数组
simple_array_data = ["苹果", "香蕉", "橙子", "葡萄"]
with open('simple_array.json', 'w', encoding='utf-8') as f:
    json.dump(simple_array_data, f, ensure_ascii=False, indent=2)

print("5. 处理简单数组:")
loader5 = JSONLoader(
    file_path="simple_array.json",
    jq_schema=".[]",  # 提取数组中的每个元素
    text_content=True
)
docs5 = loader5.load()
print(f"加载了 {len(docs5)} 个文档")
for i, doc in enumerate(docs5):
    print(f"文档 {i+1}: {doc.page_content}")
    print(f"元数据: {doc.metadata}\n")

# 测试6: 复杂嵌套数据
complex_data = {
    "products": [
        {
            "id": 1,
            "name": "笔记本电脑",
            "specs": {
                "cpu": "Intel i7",
                "ram": "16GB",
                "storage": "512GB SSD"
            },
            "price": 8999,
            "reviews": [
                {"rating": 5, "comment": "很好用"},
                {"rating": 4, "comment": "性能不错"}
            ]
        }
    ]
}

with open('complex_data.json', 'w', encoding='utf-8') as f:
    json.dump(complex_data, f, ensure_ascii=False, indent=2)

print("6. 处理复杂嵌套数据 - 提取产品名称:")
loader6 = JSONLoader(
    file_path="complex_data.json",
    jq_schema='.products[]',
    content_key="name",
    text_content=True
)
docs6 = loader6.load()
print(f"加载了 {len(docs6)} 个文档")
for i, doc in enumerate(docs6):
    print(f"文档 {i+1}: {doc.page_content}")
    print(f"元数据: {doc.metadata}\n")

print("=== JSONLoader 工作原理解释 ===")
print("""
JSONLoader的工作原理：

1. jq_schema参数：使用jq语法来查询和转换JSON数据
   - '.users[]' 表示提取users数组中的每个元素
   - '.' 表示提取整个JSON对象
   - '.users[] | {name: .name, city: .city}' 表示从每个用户中提取特定字段并重组

2. text_content参数：
   - True: 要求提取的内容必须是字符串类型
   - False: 允许提取各种数据类型（字典、列表、数字等）

3. content_key参数：
   - 当text_content=True时，指定字典中的哪个键的值作为文档内容
   - 其他字段会成为metadata

4. 输出：返回Document对象列表，每个Document包含page_content和metadata
""")


# JS0NLoader依赖jq库，通过pip installjq安装。JSONLoader使用jq的解析语法，常见如:
# .表示根、[]表示数组
# .name表示从根取name的值
# .hobby[1]表示取hobby对应数组的第二个元素
# .[]表示将数组内的每个字典(JSON对象)都取到
# .[].name表示取数组内每个字典(JSON)对象的name对应的值
#
# JS0NLoader初始化有4个主要参数:
# file_path:文件路径，必填
# jq_schema:jq解析语法，必填text_content:抽取到的是否是字符串，默认True，
# 非必填json_lines:是否是JsonLines文件，默认False，非必填
# JsonLines文件:每一行都是一个独立的字典(Json对象)