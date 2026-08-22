# v-askbot: Voice-Enabled Multilingual RAG Model
### 🌴 HackerHouse Goa 2026 Shortlisting Task 2

A voice-enabled Retrieval-Augmented Generation (RAG) system built for **HackerHouse Goa 2026**, with Sarvam speech-to-text, five chunking strategies, in-memory hybrid retrieval, Groq answer generation, structured orchestration, and groundedness guardrails.

---

## ⚡ Key Highlights & Technical Requirements

| Requirement | Implementation in `v-askbot` |
|---|---|
| **1. Speech-to-Text** | **Sarvam AI** (`saaras:v3` API); the browser records audio and uploads it to the backend |
| **2. Vast Chunking** | **5 distinct strategies**: Recursive Semantic, Metadata-Aware, Parent-Document Hierarchical, Contextual Sentence-Window, Fixed-Size Overlap |
| **3. Latency Target** | In-memory NumPy retrieval is sub-millisecond in the local benchmark; external Sarvam/Groq latency is measured separately in the end-to-end report |
| **4. Latency Analytics** | Statistical profiler tracking **P50 / P70 / P100** across every bundled multilingual sample |
| **5. Model Harness** | Pydantic JSON schemas, multi-turn tool execution (`vector_search`, `language_aligner`, `grounding_verifier`), automated retries |
| **6. Guardrails** | Prompt injection shield, toxicity filter, PII redaction, relevance sufficiency thresholding, hallucination detection & refusal triggers |
| **7. Automatic Hybrid Routing** | Local MSMARCO-XI retrieval first; low-relevance queries automatically fall back to Groq Compound web search |
| **8. Multilingual Indic UX** | Native support for Hindi (हिन्दी), Bengali (বাংলা), Tamil (தமிழ்), Telugu (తెలుగు), Marathi (मराठी), Gujarati, Kannada, Malayalam, Punjabi, Odia, English |
| **9. UI Aesthetic** | HH Goa-inspired editorial interface with automatic hybrid source labeling and telemetry HUD |

---

## 🚀 Quickstart & Installation

### 1. Clone & Setup Environment
```bash
git clone https://github.com/your-username/v-askbot.git
cd v-askbot
pip install -r requirements.txt
```

### 2. Configure API Keys (`.env`)
```env
GROQ_API_KEY=gsk_your_groq_key_here
SARVAM_API_KEY=sk_your_sarvam_key_here
ONLINE_SEARCH_MODEL=groq/compound-mini
PORT=8000
```

### 3. Start the Server
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000`** in your browser to interact with the voice interface!

---

## 📊 Latency Analytics (P50 / P70 / P100)

Run the automated statistical benchmark suite:
```bash
python benchmark_runner.py
```

The benchmark reports measured values only and marks the 200 ms target as passed or failed. In the current environment, the latest 66-run text-pipeline measurement was:

- **P50**: `279.8 ms`
- **P70**: `1264.0 ms`
- **P100**: `9406.8 ms`
- **Under target**: `0.0%`

The retrieval and guardrail portions are fast; the remaining latency is dominated by external translation/model services. Do not submit these numbers as passing SLO evidence. Configure valid production API keys and rerun the benchmark before submission.

The repository currently includes a small multilingual sample at `data/msmarco_indic_samples.json`. To benchmark an exported full dataset, set `DATA_PATH` in `.env` to its JSON path.

## Automatic Hybrid Routing

Every query follows one route automatically:

1. Search the local MSMARCO-XI index.
2. If retrieval relevance is below the configured threshold, call the Groq Compound web-search model.
3. If web search is unavailable, return the verified-context refusal.

Online answers are labeled `ONLINE_SEARCH_RESULT_UNVERIFIED` because they are not grounded in the local dataset. Configure `GROQ_API_KEY` and use a Groq account/model with Compound web search access.

---

## 🧩 Vast Chunking Strategies

1. **Recursive Semantic Structure**: Splits on natural Indic (`।`, `?`) and Latin (`.`, `!`) sentence boundaries without breaking grammatical meaning.
2. **Metadata-Aware Contextual**: Injects domain taxonomy, language ISO tag, and MSMARCO-XI provenance lineage into every chunk.
3. **Parent-Document Hierarchical**: Generates fine-grained sentence-level child vectors for pinpoint vector similarity, while providing the full parent passage for LLM generation context.
4. **Contextual Sentence-Window**: Embeds the central focal sentence enriched with a sliding window of peripheral sentences.
5. **Fixed-Size Overlap (Baseline)**: Standard 150-word chunks with 30-word sliding overlap for baseline benchmarking.

---

## 🛡️ Model Harness & Guardrails

- **Input Guardrails**: Protects against jailbreak patterns, prompt injections, toxic content, and redacts PII emails/phones.
- **Retrieval Guardrails**: Discards low-relevance matches (< 0.25 similarity score) to prevent grounded hallucinations.
- **Output Groundedness Guardrail**: Performs n-gram and entity overlap verification between generated answer and retrieved context. Triggers standard refusal when ungrounded.
- **Refusal Trigger**: *"I do not have sufficient verified context in the MSMARCO-XI dataset to answer this question accurately."* (Localized in each Indic language).

---

## 📹 Video Submission Guidelines (#RAGInGoa)

- **Video 1 (Process)**: 90-second video demonstrating team workflow and development.
- **Video 2 (Demo)**: End-to-end voice query demonstration showing live transcription, sub-200ms latency HUD, and multilingual answers.
- **Social Tags**: Every team member must post both videos on Instagram & X with **`#RAGInGoa`**.
