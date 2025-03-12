# GraphragKM - GraphRAG 驱动的 AI 本体生成工具

GraphragKM 是一款基于 GraphRAG 的 AI 本体生成工具，能够从 PDF 文档中自动提取知识，并生成 OWL 本体和 UML 模型。它整合了文本提取、OCR 识别、图谱构建、推理等技术，为用户提供一站式知识图谱和本体生成解决方案。

## 功能特点

- **PDF 文档处理和文本提取**：支持从 PDF 文档中提取文本，获取关键信息。
- **图像 OCR 识别**：支持图像中的文本提取，帮助识别扫描文档或图片中的内容。
- **基于 GraphRAG 的知识图谱构建**：自动构建知识图谱，将知识以图谱的形式进行可视化。
- **实体和关系推理**：从提取的文本和图像中推理出实体及其关系，构建更完整的知识图谱。
- **自动生成 OWL 本体**：根据提取的信息自动构建 OWL 本体，支持语义推理和知识存储。
- **自动生成 StarUML 类图**：将本体结构转换为 UML 类图，方便可视化理解和编辑。

## 安装[README.md](README.md)

```bash
pip install GraphragKM
```

## 使用方法

### 命令行使用

```bash
# 交互式运行
graphragkm run

# 指定输入文件
graphragkm run -i input.pdf
```

### 生成文件

运行后，程序会在当前目录的 output 文件夹下生成以下文件：

- `ontology.owl`：生成的 OWL 本体文件。
- `uml_model.uml`：UML 类图文件（StarUML 格式）。

### 配置

首次运行时，程序会在当前目录创建`config.yaml`配置文件模板。您需要编辑此文件，填入正确的 API 密钥和其他配置信息。

```yaml
api:
  # Mineru API settings
  mineru_upload_url: "https://mineru.net/api/v4/file-urls/batch"
  mineru_results_url_template: "https://mineru.net/api/v4/extract-results/batch/{}"
  mineru_token: "YOUR_MINERU_TOKEN"

  # Chat model settings
  chat_model_api_key: "YOUR_CHAT_MODEL_API_KEY"
  chat_model_api_base: "https://api.deepseek.com"
  chat_model_name: "deepseek-chat"

  # Embedding model settings
  embedding_model_api_key: "YOUR_EMBEDDING_MODEL_API_KEY"
  embedding_model_api_base: "https://open.bigmodel.cn/api/paas/v4/"
  embedding_model_name: "embedding-3"

app:
  # OWL Namespace
  owl_namespace: "https://example.com/"

  # Maximum concurrent requests
  max_concurrent_requests: 25
```

## 依赖项

- Python 3.11+
- graphrag
- easyocr
- openai
- pandas
- rdflib
- rich
- click
- scikit-learn
- 完整依赖项请参见 pyproject.toml
