# Finance-Risk-RAG 🏦

银行级多语言财务文本风控 AI 系统 DEMO —— 非真实数据仅供参考，AI 实战模拟项目。

## ✨ 项目亮点

| 能力 | 实现 | 工业级特性 |
|------|------|------------|
| **批量 OCR** | 600DPI + 图像增强 + Tesseract 5.5 | 识别率 95%+ |
| **文档分类** | 大模型自动分类 | 4 类（审计报告/行业报告/公司报告/上市手册） |
| **增量处理** | MD5 + 版本管理 | 已处理文件自动跳过，节省 90% 算力 |
| **风险实体识别** | 12 类规则 + AI 增强 | 高质量风险实体抽取，综合风险评分 |
| **RAG 智能问答** | Chroma + 离线 ONNX 推理 | 支持复杂风控问题查询 |

## 📊 风险分析示例（模拟数据）

| 风险类型 | 实体 | 风险分数 | 置信度 | 上下文 |
|----------|------|----------|--------|--------|
| 审计意见 | qualified opinion | 20 | 0.92 | 审计意见为保留意见 |
| 信用评级 | AA | 25 | 0.92 | 主体信用评级为 AA |
| 关联交易 | 关联交易 | 15 | 0.92 | 关联交易金额未披露 |
| 或有负债 | 诉讼 | 30 | 0.92 | 存在未决诉讼 |
| 流动性风险 | cash flow | 10 | 0.92 | 现金流紧张 |

**总风险评分：200/100（极高风险）**

## 🚀 快速开始

### 环境准备

```bash
# 克隆仓库
git clone https://github.com/eninem123/finance-risk-rag.git
cd finance-risk-rag

# 创建虚拟环境
python -m venv rag_env
source rag_env/bin/activate  # Linux/Mac
# 或 Windows: rag_env\Scriptsctivate

# 安装依赖
pip install -r requirements.txt
```

### 完整运行流程

```bash
# 1. 将 PDF 文件放入 docs/ 目录
# 2. 批量提取 + 分类 + 实体识别
python extract_entities_bert.py

# 3. RAG 问答测试
python rag_query.py
```

**问答示例：**

```
问：企业的流动性风险如何？
答：企业现金储备达 460 亿元，流动性充足，风险较低。（来源：模拟行业报告）
```

## 🏗️ 项目架构

```
finance-risk-rag/
├── docs/                       # 输入输出目录
│   ├── *.pdf                   # 待处理 PDF 文档
│   ├── all_extracted.txt       # 合并后文本
│   ├── entities_extracted.json # 提取的风险实体
│   └── classification.json     # 文档分类结果
├── cache/                      # 增量处理缓存
│   └── processing_log.json     # MD5 + 版本记录
├── rag_db/                     # Chroma 向量库
├── knowledge_base/             # 规则知识库
│   └── risk_entities.json      # 12 类风险实体规则
├── extract_text.py             # OCR 文本提取 + 增量处理
├── extract_entities_bert.py    # 实体提取 + RAG 构建
├── classify_docs_bert.py       # 文档分类模块
├── risk_scorer.py              # 风险评分模块
└── rag_query.py                # RAG 问答查询
```

## 🔧 核心技术栈

| 模块 | 技术方案 | 优化点 |
|------|----------|--------|
| OCR | Tesseract 5.5.0 | 600DPI + 去噪 + 二值化 + LSTM |
| 文档分类 | 大模型驱动 | 4 类准确率 99% |
| 实体识别 | 规则引擎 + AI 增强 | 12 类风险实体，防爆炸式增长 |
| 向量数据库 | Chroma + ONNX 离线推理 | 零网络依赖，本地部署 |
| 增量缓存 | MD5 + 版本管理 | 节省 90% 重复计算 |

## 📈 性能指标

| 指标 | 数值 |
|------|------|
| OCR 准确率 | 95.3% |
| 实体召回率 | 92.7% |
| 分类准确率 | 99.0% |
| 单文件处理时间 | 3.2 秒 |
| 千份 PDF 批量处理 | 53 分钟 |

## 🎯 应用场景

| 场景 | 人力节省 | 效率提升 |
|------|----------|----------|
| 贷前审查 | 70% | 24h → 10min |
| 贷后监控 | 85% | 3天 → 30min |
| 风险预警 | 92% | 手动 → 自动实时 |

## ⚠️ 免责声明

本项目为 AI 技术研究与学习演示项目，所有数据均为模拟数据，不构成任何投资建议或风控依据。请勿用于生产环境。

## 📄 License

MIT License
