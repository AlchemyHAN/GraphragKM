# GraphragKM - AI Ontology Generation Tool Driven by GraphRAG

GraphragKM is an AI ontology generation tool based on GraphRAG that can automatically extract knowledge from PDF documents and generate OWL ontologies and UML models. It integrates text extraction, OCR recognition, graph construction, inference, and other technologies to provide users with a one-stop knowledge graph and ontology generation solution.

## Features

- **PDF Document Processing and Text Extraction**: Supports extracting text from PDF documents to obtain key information.
- **Image OCR Recognition**: Supports text extraction from images, helping recognize content from scanned documents or pictures.
- **GraphRAG-Based Knowledge Graph Construction**: Automatically constructs knowledge graphs and visualizes knowledge in graph form.
- **Entity and Relationship Inference**: Infers entities and their relationships from extracted text and images to build a more complete knowledge graph.
- **Automatic OWL Ontology Generation**: Automatically constructs OWL ontology from extracted information, supporting semantic reasoning and knowledge storage.
- **Automatic StarUML Class Diagram Generation**: Converts ontology structures into UML class diagrams for easy visualization and editing.

## Installation

```bash
pip install GraphragKM
```

## Usage

### Command Line Usage

```bash
# Interactive run
graphragkm run

# Specify input file
graphragkm run -i input.pdf
```

### Generated Files

After execution, the program will generate the following files in the `output` folder in the current directory:

- `ontology.owl`: The generated OWL ontology file.
- `uml_model.puml`: The UML class diagram file (StarUML format).

### Configuration

On the first run, the program will create a `config.yaml` configuration file template in the current directory. You need to edit this file and fill in the correct API keys and other configuration information.

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

## Dependencies

- Python 3.11+
- graphrag
- easyocr
- openai
- pandas
- rdflib
- rich
- click
- scikit-learn
- For a full list of dependencies, see `pyproject.toml`
