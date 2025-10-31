# classify_docs_bert.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from transformers import BertTokenizer, BertModel

model_path = "./chinese-bert-wwm-ext"  # 替换为你本地文件夹的实际路径
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)
# 加载中文 BERT 分类模型
#tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
#model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-bert-wwm-ext", num_labels=4)

# 类别映射
label_map = {0: "审计报告", 1: "行业报告", 2: "公司研究报告", 3: "上市手册"}

def classify_text(text):
    inputs = tokenizer(text[:512], return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = label_map[pred.argmax().item()]
    confidence = pred.max().item()
    return {"type": label, "confidence": confidence}

# 测试
if __name__ == "__main__":
    with open("docs/all_extracted.txt", "r", encoding="utf-8") as f:
        text = f.read()
    result = classify_text(text)
    print(f"分类结果：{result}")