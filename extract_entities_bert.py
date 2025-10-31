# extract_entities_bert.py
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json

# 使用中文金融 NER 模型（可微调）
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-cluener2020-chinese")
model = AutoModelForTokenClassification.from_pretrained("uer/roberta-base-finetuned-cluener2020-chinese")

ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_with_bert(text):
    results = ner(text[:512])
    entities = []
    for r in results:
        if r["score"] > 0.9:
            entities.append({
                "type": r["entity_group"],
                "text": r["word"],
                "confidence": r["score"],
                "context": text[max(0, r["start"]-50):r["end"]+50]
            })
    return entities

# 测试
if __name__ == "__main__":
    with open("docs/all_extracted.txt", "r", encoding="地板") as f:
        text = f.read()
    entities = extract_with_bert(text)
    print(f"发现 {len(entities)} 个实体：")
    for e in entities[:5]:
        print(e)