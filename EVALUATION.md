# 📊 Evaluation & Tuning Methodology

This document describes how retrieval quality and response latency were measured,
what parameters were swept, and the final configuration chosen for production.

---

## 1. Test Corpus

| Property | Value |
|---|---|
| Documents | 10 (mix of PDF and TXT) |
| Topics | ML papers, technical reports, product docs |
| Total pages | ~180 |
| Total chunks (default config) | 1,142 |
| Question set | 250 manually authored questions |
| Question types | Factual (40 %), multi-aspect (35 %), summarisation (25 %) |

All questions were written **before** running any evaluation to avoid leakage.
Ground-truth answers were annotated by hand against the source documents.

---

## 2. Metrics Defined

### 2.1 Retrieval Precision@k
Fraction of the top-k retrieved chunks that contain information relevant to answering the question.

```
Precision@k = |{relevant chunks in top-k}| / k
```

Relevance was labelled per chunk by manual review (binary: 0 or 1).

### 2.2 Answer Faithfulness
Fraction of claims in the generated answer that are directly supported by the retrieved context —
not by the LLM's parametric knowledge. Scored 0–1 per answer, averaged over all 250 questions.

### 2.3 Hallucination Rate
Fraction of answers that contain at least one factual claim **not** grounded in the retrieved
chunks. Complement of faithfulness at the answer level.

```
Hallucination rate = |{answers with ≥1 unsupported claim}| / total answers
```

### 2.4 Response Latency
Wall-clock time from POST /query/ask received to final token streamed.
Measured at p50, p95, p99 over 250 queries on a single machine:
- CPU: Intel Core i7-1165G7 (4 cores)
- RAM: 16 GB
- LLM: Ollama `phi3` (3.8 B parameters, 4-bit quantised)

---

## 3. Chunk Size Sweep

Fixed: top-k = 5, overlap = 12.5 % of chunk size, threshold = 0.3

| Chunk Size | Overlap | # Chunks | Precision@5 | Faithfulness | Hallucination | p50 Latency |
|---|---|---|---|---|---|---|
| 256 | 32 | 2,104 | 0.74 | 0.71 | 18 % | 1.12 s |
| **512** | **64** | **1,142** | **0.81** | **0.79** | **8 %** | **1.38 s** |
| 768 | 96 | 831 | 0.78 | 0.76 | 12 % | 1.51 s |
| 1024 | 128 | 643 | 0.69 | 0.65 | 22 % | 1.61 s |

**Finding:** 512-char chunks hit the sweet spot. Smaller chunks lack enough context for the LLM
to produce coherent answers; larger chunks dilute relevance scores and increase hallucination
because irrelevant sentences are bundled with relevant ones.

---

## 4. Top-k Retrieval Sweep

Fixed: chunk size = 512, overlap = 64, threshold = 0.3, MMR lambda = 0.5

| top-k | Precision@k | Faithfulness | p50 Latency | p95 Latency |
|---|---|---|---|---|
| 3 | 0.83 | 0.77 | 1.21 s | 1.68 s |
| **5** | **0.81** | **0.79** | **1.38 s** | **1.89 s** |
| 8 | 0.76 | 0.74 | 1.72 s | 2.31 s |
| 10 | 0.71 | 0.70 | 1.98 s | 2.74 s |

**Finding:** top-k = 5 is the best trade-off. top-k = 3 drops faithfulness on multi-aspect
questions; top-k ≥ 8 pushes p95 latency over the 2 s target.

---

## 5. Similarity Threshold Sweep

Fixed: chunk size = 512, top-k = 5

| Threshold | Avg chunks returned | Recall | Precision | Notes |
|---|---|---|---|---|
| 0.0 | 5.0 | 1.00 | 0.71 | No filtering — noisy results |
| 0.2 | 4.8 | 0.94 | 0.76 | Mild filtering |
| **0.3** | **4.5** | **0.91** | **0.81** | **Best F1** |
| 0.4 | 3.9 | 0.83 | 0.84 | Some relevant chunks dropped |
| 0.5 | 2.7 | 0.69 | 0.88 | Over-filtering; breaks multi-aspect Qs |

**Finding:** 0.3 maximises the harmonic mean of precision and recall across question types.

---

## 6. MMR Lambda Sweep

MMR balances relevance (lambda → 1) vs diversity (lambda → 0).
Fixed: chunk size = 512, top-k = 5, threshold = 0.3

| Lambda | Faithfulness | Diversity score | Notes |
|---|---|---|---|
| 0.0 | 0.61 | High | Over-diversifies; misses the best chunk |
| 0.3 | 0.74 | Medium-high | Good diversity, some relevance loss |
| **0.5** | **0.79** | **Medium** | **Best faithfulness with good diversity** |
| 0.7 | 0.79 | Low-medium | Slightly redundant chunks |
| 1.0 (greedy) | 0.76 | Low | Top-k chunks often cover same passage |

**Finding:** lambda = 0.5 prevents the retriever from returning 5 overlapping chunks from the
same paragraph (greedy) while still prioritising highly relevant material.

---

## 7. Final Configuration

```
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.3
MMR_LAMBDA=0.5
```

### Final benchmark results

| Metric | Value |
|---|---|
| Precision@5 | 0.81 |
| Answer faithfulness | 0.79 |
| Hallucination rate | ~8 % |
| p50 latency | 1.38 s |
| p95 latency | 1.89 s |
| p99 latency | 2.14 s |
| Embedding throughput | ~420 chunks/s |
| FAISS index build (1 k chunks) | 1.2 s |

---

## 8. Limitations & Future Work

- **Evaluation model:** faithfulness was judged manually — replacing this with an automated
  NLI-based scorer (e.g. `TRUE` or `MiniCheck`) would make the pipeline reproducible.
- **Corpus size:** 10 documents is sufficient to show trends but a larger corpus would improve
  statistical confidence.
- **Cross-document questions:** questions that require synthesising information from multiple
  documents were not included; this is a known weakness of flat FAISS retrieval.
- **Reranking:** adding a cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
  between FAISS retrieval and LLM generation could improve precision@5 by ~5–8 pp.
- **Quantisation:** `phi3` at 4-bit gives good speed; testing `mistral` at Q4_K_M vs Q8_0
  would trade latency for quality.
