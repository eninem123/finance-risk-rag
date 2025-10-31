# rag_query_bert.py
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI

# 使用 BERT 嵌入（比 ONNX 更准）
embedding_function = SentenceTransformerEmbeddingFunction(model_name="shibing624/text2vec-base-chinese")

chroma = chromadb.PersistentClient(path="rag_db_bert")
collection = chroma.get_or_create_collection("risk_entities", embedding_function=embedding_function)

def ask_bert(question):
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n".join([d for d in results["documents"][0]])
    
    client = OpenAI(api_key="sk-...", base_url="https://api.moonshot.cn/v1")
    prompt = f"根据以下内容回答：\n{context}\n问题：{question}"
    resp = client.chat.completions.create(model="moonshot-v1-8k", messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content

# 测试
print(ask_bert("小鹏汽车现金储备多少？"))