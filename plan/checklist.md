# AI Learning Roadmap — Weekly Checklist (Engineer-Optimized, 2026)

Audience:
- AI beginner
- Strong backend / cloud / infrastructure experience
- Goal: become an **AI systems engineer**, not a researcher

Assumed stack alignment:
- Backend: Java / Spring Boot OR Python / FastAPI
- Cloud: GCP / AWS
- Infra: Docker, Kubernetes, Terraform
- Data: SQL, Hive / BigQuery / Spark
- CI/CD: GitHub Actions

---

## 🗓️ 12-Month Weekly Checklist

────────────────────────────
## Phase 0 — Minimal Foundations (Weeks 1–3)
────────────────────────────

### Week 1
- [✅] Install Python (conda / pyenv)
- [✅] Learn NumPy basics (arrays, broadcasting)
- [ ] Learn Pandas basics (DataFrame, filtering)
- [ ] Plot simple graphs with Matplotlib
- [ ] Understand: *What is a feature?*

### Week 2
- [ ] Linear algebra intuition (vectors, dot product)
- [ ] Derivatives & gradients (conceptual)
- [ ] What is a loss function?
- [ ] Implement linear regression **from scratch**

### Week 3
- [ ] Probability basics (mean, variance)
- [ ] Understand train/test split
- [ ] Data preprocessing pipeline
- [ ] Mini-project: data cleaning + simple model

✅ Exit: You understand the **language of ML**

---

────────────────────────────
## Phase 1 — Machine Learning Fundamentals (Weeks 4–8)
────────────────────────────

### Week 4
- [ ] Supervised vs unsupervised learning
- [ ] Linear & logistic regression (sklearn)
- [ ] Bias vs variance intuition

### Week 5
- [ ] Decision trees
- [ ] Random Forest, Gradient Boosting
- [ ] Overfitting & regularization

### Week 6
- [ ] Model evaluation metrics
- [ ] Confusion matrix, ROC-AUC
- [ ] Cross-validation

### Week 7
- [ ] Feature engineering
- [ ] Pipelines in sklearn
- [ ] Error analysis

### Week 8
- [ ] Build ML service:
  - FastAPI
  - sklearn model
  - REST endpoint
- [ ] Add logging & basic monitoring

✅ Exit: You can **build & deploy classical ML systems**

---

────────────────────────────
## Phase 2 — Deep Learning Essentials (Weeks 9–16)
────────────────────────────

### Week 9
- [ ] What is a neural network?
- [ ] Perceptron & MLP
- [ ] Activation functions

### Week 10
- [ ] Backpropagation (intuition)
- [ ] Gradient descent variants
- [ ] Vanishing/exploding gradients

### Week 11
- [ ] CNNs
- [ ] Image classification basics
- [ ] Data augmentation

### Week 12
- [ ] RNNs & LSTMs
- [ ] Sequence modeling
- [ ] Why RNNs struggle

### Week 13
- [ ] Transformers (high-level)
- [ ] Attention mechanism
- [ ] Why Transformers won

### Week 14
- [ ] PyTorch training loops
- [ ] GPU vs CPU trade-offs
- [ ] Experiment tracking

### Week 15
- [ ] Model serving patterns
- [ ] Batch vs online inference

### Week 16
- [ ] Deploy DL model:
  - Docker
  - FastAPI
  - GPU (optional)

✅ Exit: You understand **how deep learning actually works**

---

────────────────────────────
## Phase 3 — LLMs & Generative AI (Weeks 17–24)
────────────────────────────

### Week 17
- [ ] Tokens & tokenization
- [ ] Context windows
- [ ] Cost & latency implications

### Week 18
- [ ] Prompt engineering patterns
- [ ] Few-shot vs zero-shot
- [ ] Output control (JSON schemas)

### Week 19
- [ ] Embeddings
- [ ] Semantic search
- [ ] Vector databases (FAISS)

### Week 20
- [ ] RAG architecture
- [ ] Chunking strategies
- [ ] Retrieval evaluation

### Week 21
- [ ] Build RAG service:
  - Document ingestion
  - Vector store
  - Query pipeline

### Week 22
- [ ] Streaming responses
- [ ] Error handling & retries
- [ ] Guardrails

### Week 23
- [ ] Fine-tuning concepts
- [ ] LoRA / adapters
- [ ] When NOT to fine-tune

### Week 24
- [ ] Full LLM app:
  - Auth
  - Rate limiting
  - Cost tracking

✅ Exit: You can **build serious LLM applications**

---

────────────────────────────
## Phase 4 — AI in Production (Weeks 25–36)
────────────────────────────

### Week 25
- [ ] ML lifecycle in production
- [ ] Offline vs online evaluation

### Week 26
- [ ] Experiment tracking (MLflow / W&B)
- [ ] Model versioning

### Week 27
- [ ] Monitoring LLMs
- [ ] Hallucination detection
- [ ] Quality metrics

### Week 28
- [ ] CI/CD for ML
- [ ] Canary deployments

### Week 29
- [ ] Docker optimization
- [ ] Kubernetes basics for ML

### Week 30
- [ ] Autoscaling inference
- [ ] GPU scheduling basics

### Week 31
- [ ] Cost optimization strategies
- [ ] Model routing & fallbacks

### Week 32–36
- [ ] Production AI system:
  - Observability
  - Rollback
  - Security
  - Compliance

✅ Exit: You can **operate AI systems safely**

---

────────────────────────────
## Phase 5 — Agents & AI Systems (Weeks 37–52)
────────────────────────────

### Week 37
- [ ] What are AI agents?
- [ ] ReAct pattern

### Week 38
- [ ] Tool calling
- [ ] Function routing

### Week 39
- [ ] Planning & task decomposition
- [ ] Memory systems

### Week 40
- [ ] Multi-agent systems
- [ ] Coordination patterns

### Week 41
- [ ] Failure recovery
- [ ] Self-reflection loops

### Week 42–52
- [ ] Capstone project:
  - Multi-agent workflow
  - Tool usage
  - Logging & evaluation
  - Cloud deployment

✅ Exit: You are an **AI systems engineer**

---

# ☁️ Backend / Cloud Stack Alignment

| Layer | Recommended Tools |
|-----|------------------|
| API | FastAPI / Spring Boot |
| ML | scikit-learn, PyTorch |
| LLM | OpenAI / HF / Bedrock |
| Data | SQL, BigQuery, Hive |
| Vector DB | FAISS, Pinecone |
| Infra | Docker, Kubernetes |
| CI/CD | GitHub Actions |
| Cloud | GCP Vertex AI / AWS |

---

# 🗺️ Visual Roadmap Diagram

- **Foundations** → **Machine Learning** → **Deep Learning** → **LLMs & RAG** → **Production AI** → **Agents & Systems**

- **Notes:** each stage contains 4–8 weeks of focused work (concepts → projects → deployable demo).