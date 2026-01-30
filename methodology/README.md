# Methodology

## Overview

Our approach uses a systematic **2-Tier Retrieval-Augmented Generation (RAG)** pipeline combined with **real technique execution** to produce evidence-based, actionable fairness interventions.

---

## The 2-Tier RAG System

### Tier 1: Discovery

Systematically retrieve relevant fairness techniques from:
- 72+ catalogued intervention techniques
- Academic literature (300+ citations)
- Industry best practices
- Regulatory guidance (EU AI Act, etc.)

**Retrieval Method**: Hybrid semantic search combining:
- Vector similarity (OpenAI embeddings)
- Metadata filtering (intervention stage, fairness metric, complexity)

### Tier 2: Synthesis

Consolidate Tier 1 discoveries into actionable deliverables using LLM synthesis:
- Integration workflows
- Implementation guides
- Validation frameworks
- Executive presentations

---

## Execution Pipeline

### Stage 1: Baseline Measurement
```
Load Data -> Train Baseline Model -> Measure Fairness Metrics
```

### Stage 2: Intervention Application
```
Pre-Processing -> In-Processing -> Post-Processing
     |                |                |
     v                v                v
  Data              Model            Decisions
Transform         Constraints        Threshold
                                    Calibration
```

### Stage 3: Validation
```
Permutation Tests -> Bootstrap CIs -> Cross-Validation
       |                  |                |
       v                  v                v
  Statistical        Confidence       Robustness
 Significance        Intervals          Check
```

---

## Documents

- [RAG System Architecture](./rag-system.md)
- [Fairness Metrics Definitions](./fairness-metrics.md)
- [Validation Framework](./validation-framework.md)

---

## Navigation

- [Back to Main Portfolio](../)
- [Case Studies](../case-studies/)
- [Services](../services/)
