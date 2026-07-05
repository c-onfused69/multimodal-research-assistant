# 🧠 Multimodal Research Assistant

An advanced, production-ready Agentic RAG system featuring Multimodal Late-Interaction Retrieval (ColPali), GraphRAG, and adaptive autonomous research workflows.

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-00a393)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_Store-blueviolet)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflows-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Key Features

- **Multi-Format Ingestion**: Native handling of PDFs, Images, Audio, and text using specialized parsers (Docling, PDFium, GPT-4V, Whisper).
- **Advanced Retrieval Strategies**:
  - **Hybrid Search**: Fuses Dense (bge-m3) and Sparse (BM25/SPLADE) search results using Reciprocal Rank Fusion (RRF).
  - **Visual RAG**: Late-interaction visual embeddings (ColPali) for screenshot-level retrieval.
  - **GraphRAG**: NetworkX-based multi-hop relationship extraction and search.
- **Agentic Workflows**: Powered by `LangGraph`, featuring state-machine routing for direct Q&A, Web Search (Tavily), SQL execution, and "Deep Research" synthesizing multi-step analysis.
- **Production Guardrails**: Real-time input scanning (LlamaGuard/Regex for PII and Prompt Injection) and hallucination output verification.
- **Enterprise Infrastructure**: Integrated semantic caching (Redis), API tracing (Langfuse), and CI/CD evaluation gates (DeepEval).

---

## 🛠️ Tech Stack

| Component               | Technology         | Description                                      |
| :---------------------- | :----------------- | :----------------------------------------------- |
| **Backend Framework**   | FastAPI            | High-performance async API server                |
| **Frontend**            | Streamlit          | Chat-based User Interface                        |
| **Vector Database**     | Qdrant             | Storage for Text and ColPali visual multivectors |
| **Cache & State**       | Redis              | Semantic caching & manifest tracking             |
| **Agent Orchestration** | LangGraph          | State-based LLM workflows                        |
| **LLMs / Vision**       | OpenAI / Anthropic | Core text generation and visual captioning       |
| **Observability**       | Langfuse           | E2E trace logging and user session evaluation    |

---

## 🚀 Getting Started / Installation

Follow these steps to spin up the project locally.

### 1. Prerequisites

Ensure you have the following installed:

- **Python 3.12+**
- **Docker** and **Docker Compose**
- **Git**

### 2. Clone the Repository

```bash
git clone https://github.com/c-onfused69/multimodal-research-assistant.git
cd "multimodal-research-assistant"
```

### 3. Spin Up Infrastructure (Qdrant & Redis)

The system relies on Qdrant and Redis. Start them in the background using Docker Compose:

```bash
docker-compose up -d
```

### 4. Install Dependencies

Install the required packages. You can use standard `pip` to install the project along with its optional capabilities:

```bash
# Install the core app along with all extra tools (UI, multimodal models, graphs, etc)
pip install -e ".[multimodal,eval,dev,graph,finetune,tools,ui]"
```

### 5. Configure Environment Variables

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Open `.env` and configure your API keys (e.g., `OPENAI_API_KEY`, `TAVILY_API_KEY`, and `LANGFUSE_PUBLIC_KEY`).

### 6. Run the Application

You need two separate terminal windows—one for the backend API and one for the frontend UI.

**Terminal 1: Start the FastAPI Backend**

```bash
uvicorn api.main:app --reload --port 8000
```

**Terminal 2: Start the Streamlit UI**

```bash
streamlit run ui/app.py
```

> The UI will automatically open in your default browser at `http://localhost:8501`.

---

## 🧪 Usage Examples

### Running the Data Ingestion Pipeline

To ingest your research documents (PDFs, markdown, text files), place them in a folder (e.g., `./data`) and run the pipeline:

```bash
python -m ingestion.pipeline --source ./data
```

This script will parse, chunk, embed, and load the documents into Qdrant.

### Automated Evaluation

To test the agent against the golden dataset (to ensure no regressions):

```bash
python -m eval.run_eval --dataset eval/golden_dataset/questions.jsonl
```

---

## 📫 Contact / Connect

- **Author**: Multimodal AI Team
- **Issues**: [Report a bug](https://github.com/c-onfused69/multimodal-research-assistant/issues)
- **Contributions**: Pull requests are warmly welcomed! Please refer to our contribution guidelines for major architectural shifts.

> [!TIP]
> **New to Agentic RAG?** Start by running simple queries in the Streamlit UI set to "Fast Mode" to see hybrid search in action, then toggle to "Deep Mode" to observe the LangGraph agent synthesize multi-step research.
