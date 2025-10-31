# rag_query.py
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from openai import OpenAI

client = OpenAI(api_key="sk-VjTuyd0GTdRA2Wi1HvJruk3WRhJwDWrPvUwl4TpyvRSjXs94", base_url="https://api.moonshot.cn/v1")
embedding_function = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
chroma = chromadb.PersistentClient(path="rag_db")
collection = chroma.get_collection("risk_entities", embedding_function=embedding_function)

def ask(question):
    results = collection.query(query_texts=[question], n_results=4)
    context = "\n\n".join([
        f"【{m['type']}】{m['text']}\n上下文：{d.split('上下文：')[-1].strip()}"
        for d, m in zip(results["documents"][0], results["metadatas"][0])
    ])
    
    prompt = f"""
根据以下风险实体，简洁回答（<80字）：

实体：
{context}

问题：{question}
"""
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    q = "你这里有什么结论"
    print(f"问：{q}")
    print(f"答：{ask(q)}")