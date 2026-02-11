from langchain_community.document_loaders import TextLoader
import os

# 首先创建几个测试文本文件
test_content1 = """这是第一个测试文件的内容。
LangChain是一个强大的框架。
它可以帮助我们构建语言模型应用。
今天我们要学习TextLoader的用法。"""

test_content2 = """这是第二个测试文件的内容。
TextLoader用于加载纯文本文件。
它可以处理单个文件或多个文件。
支持不同的编码格式。"""

test_content3 = """姓名：张三
年龄：25岁
城市：北京
职业：软件工程师
技能：Python, JavaScript, SQL"""

# 创建测试文件 - 使用UTF-8 BOM格式（更兼容）
with open('test1.txt', 'w', encoding='utf-8-sig') as f:
    f.write(test_content1)

with open('test2.txt', 'w', encoding='utf-8-sig') as f:
    f.write(test_content2)

with open('test3.txt', 'w', encoding='utf-8-sig') as f:
    f.write(test_content3)

print("=== LangChain TextLoader 完全指南 ===\n")

# 测试1: 基本用法
print("1. 基本用法 - 加载单个文本文件:")
loader1 = TextLoader(file_path="test1.txt", encoding='utf-8-sig')
documents1 = loader1.load()
print(f"加载了 {len(documents1)} 个文档")
for i, doc in enumerate(documents1):
    print(f"文档 {i+1} 内容:\n{doc.page_content}")
    print(f"元数据: {doc.metadata}\n")






# 测试2: 使用autodetect_encoding参数（推荐）
print("2. 启用编码自动检测:")
loader2 = TextLoader(file_path="test1.txt", autodetect_encoding=True)
documents2 = loader2.load()
print(f"加载了 {len(documents2)} 个文档")
for i, doc in enumerate(documents2):
    print(f"文档 {i+1} 内容预览: {doc.page_content[:50]}...")
    print(f"元数据: {doc.metadata}\n")





# 测试3: 使用明确的UTF-8编码
print("3. 明确指定UTF-8编码:")
loader3 = TextLoader(file_path="test1.txt", encoding='utf-8-sig')
documents3 = loader3.load()
print(f"加载了 {len(documents3)} 个文档")
for i, doc in enumerate(documents3):
    print(f"文档 {i+1} 内容预览: {doc.page_content[:50]}...")
    print(f"元数据: {doc.metadata}\n")





# 测试4: 模拟不同编码文件 (创建GBK编码文件)
gbk_content = "这是一个GBK编码的文件\n包含中文内容\n用于测试编码检测功能"
with open('test_gbk.txt', 'w', encoding='gbk') as f:
    f.write(gbk_content)

print("4. 处理不同编码文件 (GBK):")
# 使用自动检测
loader4 = TextLoader(file_path="test_gbk.txt", autodetect_encoding=True)
documents4 = loader4.load()
print(f"使用自动检测成功加载了 {len(documents4)} 个文档")
for i, doc in enumerate(documents4):
    print(f"文档 {i+1} 内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}\n")




# 测试5: 手动指定编码
print("5. 手动指定编码读取GBK文件:")
loader5 = TextLoader(file_path="test_gbk.txt", encoding='gbk')
documents5 = loader5.load()
print(f"加载了 {len(documents5)} 个文档")
for i, doc in enumerate(documents5):
    print(f"文档 {i+1} 内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}\n")



# 测试6: 处理大文件
large_content = "这是大文件测试内容。\n" * 100  # 创建一个较大的文件
with open('large_test.txt', 'w', encoding='utf-8-sig') as f:
    f.write(large_content)

print("6. 处理大文件:")
loader6 = TextLoader(file_path="large_test.txt", encoding='utf-8-sig')
documents6 = loader6.load()
print(f"加载了 {len(documents6)} 个文档")
print(f"文档长度: {len(documents6[0].page_content)} 字符")
print(f"元数据: {documents6[0].metadata}\n")





# 测试7: 错误处理 - 修正版
print("7. 错误处理测试:")
try:
    # 尝试加载不存在的文件
    loader7 = TextLoader(file_path="nonexistent.txt")
    documents7 = loader7.load()
except RuntimeError as e:  # TextLoader会将底层异常包装成RuntimeError
    print(f"TextLoader运行时错误: {e}")
except Exception as e:
    print(f"其他错误: {e}")

# 测试8: 特殊字符处理
special_content = """特殊字符测试：
英文标点: !@#$%^&*()
中文标点: ，。！？；：""
数字: 123456789
符号: <>{}[]|\\/:*
换行符测试
制表符测试	制表符
空格测试  空格  """
with open('special_chars.txt', 'w', encoding='utf-8-sig') as f:
    f.write(special_content)

print("8. 特殊字符处理:")
loader8 = TextLoader(file_path="special_chars.txt", encoding='utf-8-sig')
documents8 = loader8.load()
print(f"加载了 {len(documents8)} 个文档")
print(f"特殊字符内容: {repr(documents8[0].page_content[:100])}...")
print(f"元数据: {documents8[0].metadata}\n")

print("=== TextLoader 参数详解 ===")
print("""
TextLoader的主要参数：

1. file_path (必需):
   - 要加载的文本文件路径
   - 可以是单个文件路径

2. encoding (可选):
   - 文件编码格式，默认为None
   - 常见编码: 'utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1'
   - 推荐使用 'utf-8-sig' 来处理UTF-8文件

3. autodetect_encoding (可选):
   - 是否自动检测文件编码，默认False
   - 设为True时会尝试自动识别编码格式（推荐使用）

4. metadata_func (可选):
   - 自定义元数据函数，用于添加额外的元数据

最佳实践：
- 推荐使用 autodetect_encoding=True 来自动处理编码问题
- 或者明确指定正确的编码格式
- UTF-8 with BOM 使用 'utf-8-sig' 编码

错误处理：
- TextLoader会将底层异常包装成RuntimeError
- 捕获RuntimeError来处理加载错误

输出结果：
- 返回Document对象列表
- Document.page_content: 包含文件的实际文本内容
- Document.metadata: 包含源文件路径等元数据信息

使用场景：
- 加载纯文本文件
- 处理日志文件
- 读取配置文件
- 导入文档内容
""")

# 清理测试文件
test_files = ['test1.txt', 'test2.txt', 'test3.txt', 'test_gbk.txt', 'large_test.txt', 'special_chars.txt']
for file in test_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"已删除测试文件: {file}")

print("\n所有测试完成！")
