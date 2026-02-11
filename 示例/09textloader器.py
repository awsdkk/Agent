from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader(file_path="long_test_article.txt", encoding="utf-8")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 10,
    separators = ["\n\n", "\n", "。", "！", "？", "!", "?", " ", ""],
    length_function = len,
)

split_docs = text_splitter.split_documents(docs)

print(len(split_docs))

for doc in split_docs:
    print("-" * 20)
    print(doc)
    print("-" * 20)
