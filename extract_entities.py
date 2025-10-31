# extract_entities.py (终极完整版：数字识别 + 上下文增强)
import json
import re
from openai import OpenAI
import os
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from datetime import datetime
import shutil

# 初始化
client = OpenAI(api_key="sk-VjTuyd0GTdRA2Wi1HvJruk3WRhJwDWrPvUwl4TpyvRSjXs94", base_url="https://api.moonshot.cn/v1")
embedding_function = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])

# 重建向量库
if os.path.exists("rag_db"):
    shutil.rmtree("rag_db")
    print("已删除旧向量库 rag_db/")

chroma_client = chromadb.PersistentClient(path="rag_db")
collection = chroma_client.create_collection(name="risk_entities", embedding_function=embedding_function)

def load_entity_rules():
    with open("knowledge_base/risk_entities.json", "r", encoding="utf-8") as f:
        return json.load(f)

def extract_entities_rule_based(text, rules):
    entities = []
    seen = set()
    
    # 数字增强模式
    num_patterns = {
        "liquidity_risk": r'(现金储备|现金及现金等价物|cash.*reserve).*?(\d+[,\d]*\.?\d*)\s*(亿|亿元|百万|million|billion)',
        "credit_rating": r'(评级|rating).*?(AAA|AA|A|BBB|BB|B|CCC)',
        "contingent_liability": r'(诉讼|诉讼金额|pending litigation).*?(\d+[,\d]*\.?\d*)\s*(亿|万元|USD)',
        "related_transaction": r'(关联交易金额|related party).*?(\d+[,\d]*\.?\d*)\s*(亿|万元|HKD|USD)'
    }
    
    for entity_type, config in rules.items():
        for keyword in config["keywords"]:
            pattern = rf'\b{re.escape(keyword)}\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start = match.start()
                key = f"{entity_type}_{keyword}_{start}"
                if key in seen: continue
                seen.add(key)
                context = text[max(0, start-80):start+len(keyword)+80].replace("\n", " ").strip()
                entities.append({
                    "type": entity_type,
                    "text": keyword,
                    "start": start,
                    "context": context,
                    "confidence": 0.92,
                    "risk_score": config["risk_score"],
                    "description": config["description"]
                })
        
        # 数字增强
        if entity_type in num_patterns:
            for match in re.finditer(num_patterns[entity_type], text, re.IGNORECASE):
                amount = match.group(2).replace(",", "")
                unit = match.group(3) if len(match.groups()) > 2 else ""
                text_val = f"{match.group(1)}{amount}{unit}"
                start = match.start()
                key = f"{entity_type}_num_{start}"
                if key in seen: continue
                seen.add(key)
                context = match.group(0).replace("\n", " ").strip()
                entities.append({
                    "type": entity_type,
                    "text": text_val,
                    "start": start,
                    "context": context,
                    "confidence": 0.96,
                    "risk_score": config["risk_score"],
                    "description": config["description"]
                })
    return entities

def extract_entities_with_ai(text, rules):
    prompt = f"""
请提取以下12类风险实体，每类最多2个（置信>0.85），包含具体数字。

文本：
{text[:5000]}

类型：
{json.dumps({k: v["description"] for k, v in rules.items()}, ensure_ascii=False)}

输出 JSON：
{{
  "entities": [
    {{"type": "liquidity_risk", "text": "现金储备 460 亿元", "context": "小鹏汽车现金储备达460亿元...", "confidence": 0.95}}
  ]
}}
"""
    try:
        response = client.chat.completions.create(model="moonshot-v1-8k", messages=[{"role": "user", "content": prompt}], temperature=0.0)
        content = response.choices[0].message.content.strip()
        if not content: return []
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0: return []
        result = json.loads(content[start:end])
        entities = []
        for e in result.get("entities", []):
            if e.get("confidence", 0) < 0.85: continue
            t = e["type"]
            if t not in rules: continue
            e.update({"risk_score": rules[t]["risk_score"], "description": rules[t]["description"]})
            entities.append(e)
        return entities
    except Exception as e:
        print(f"AI 提取失败: {e}")
        return []

def merge_and_deduplicate(entities):
    best = {}
    for e in entities:
        t = e["type"]
        if t not in best or e["confidence"] > best[t]["confidence"]:
            best[t] = e
    return list(best.values())

def build_rag_vector_db(entities):
    if not entities:
        print("无实体，跳过向量库")
        return
    docs = [f"{e['description']}：{e['text']}\n上下文：{e['context']}" for e in entities]
    metadatas = [{k: e.get(k) for k in ["type", "text", "risk_score", "confidence"]} for e in entities]
    ids = [f"e_{i}" for i in range(len(entities))]
    collection.add(documents=docs, metadatas=metadatas, ids=ids)
    print(f"向量库构建完成！共 {len(entities)} 条实体")

if __name__ == "__main__":
    rules = load_entity_rules()
    with open("docs/all_extracted.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    print("开始实体提取（含数字识别）...")
    rule_entities = extract_entities_rule_based(text, rules)
    ai_entities = extract_entities_with_ai(text, rules)
    final_entities = merge_and_deduplicate(rule_entities + ai_entities)
    
    total_risk = sum(e["risk_score"] for e in final_entities)
    risk_level = "低风险" if total_risk < 40 else "中风险" if total_risk < 80 else "高风险"
    
    result = {
        "extracted_at": datetime.now().isoformat(),
        "total_entities": len(final_entities),
        "total_risk_score": total_risk,
        "risk_level": risk_level,
        "entities": final_entities
    }
    
    os.makedirs("docs", exist_ok=True)
    with open("docs/entities_extracted.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    build_rag_vector_db(final_entities)
    
    print(f"\n系统上线！风控 RAG 正式就绪！")
    print(f"   实体数: {len(final_entities)}")
    print(f"   总风险: {total_risk}/100 ({risk_level})")
    
    print("\nTop 5 实体（含数字）：")
    for e in final_entities[:5]:
        print(f"   {e['type']:20} | {e['text']:25} | 分数: {e['risk_score']:2} | 置信: {e['confidence']:.2f}")