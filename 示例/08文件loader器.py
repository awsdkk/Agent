import chardet


def detect_file_encoding(file_path):
    """改进的编码检测函数"""
    # 尝试多种编码格式
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'cp1252', 'iso-8859-1', 'cp936']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1000)  # 尝试读取前1000个字符
            print(f"使用 {encoding} 编码可以正常读取文件")
            return encoding
        except UnicodeDecodeError:
            continue

    # 如果上述编码都失败，使用chardet进行检测
    with open(file_path, 'rb') as file:
        raw_data = file.read(10000)
        result = chardet.detect(raw_data)
        detected_encoding = result['encoding']

    if detected_encoding:
        print(f"chardet检测到编码: {detected_encoding}")
        return detected_encoding
    else:
        print("无法确定编码，将尝试使用 'latin-1' 作为最后手段")
        return 'latin-1'


# 检测编码
encoding = detect_file_encoding("test_data.csv")
print(f"最终选择的编码: {encoding}")

# 现在使用检测到的编码加载CSV文件
from langchain_community.document_loaders import CSVLoader

try:
    loader = CSVLoader(
        file_path="test_data.csv",
        encoding=encoding,
    )

    documents = loader.load()
    print(f"成功加载了 {len(documents)} 个文档")

    for i, document in enumerate(documents[:5]):  # 只打印前5个文档作为示例
        print(f"文档 {i + 1}: {document}")

except Exception as e:
    print(f"使用编码 {encoding} 仍然失败: {str(e)}")
    print("尝试使用 'latin-1' 编码并处理特殊字符...")

    # 最后的尝试：使用latin-1编码，它能读取任何字节序列
    loader = CSVLoader(
        file_path="408.csv",
        encoding='latin-1',
    )

    documents = loader.load()
    print(f"成功加载了 {len(documents)} 个文档")

    for i, document in enumerate(documents[:5]):  # 只打印前5个文档作为示例
        print(f"文档 {i + 1}: {document}")
