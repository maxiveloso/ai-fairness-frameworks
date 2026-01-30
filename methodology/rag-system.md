# 2-Tier RAG System Architecture

## Overview

Our Retrieval-Augmented Generation system enables systematic discovery and synthesis of fairness techniques, replacing ad-hoc selection with evidence-based methodology.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TIER 1: DISCOVERY                       │
├─────────────────────────────────────────────────────────────┤
│  Query Embedding  ->  Hybrid Search  ->  Top-K Retrieval    │
│       │                    │                   │            │
│       v                    v                   v            │
│  OpenAI API          Vector DB +         Ranked            │
│  (1536 dims)         Metadata            Techniques         │
└─────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────┐
│                      TIER 2: SYNTHESIS                       │
├─────────────────────────────────────────────────────────────┤
│  Context Assembly  ->  LLM Synthesis  ->  Deliverables      │
│       │                    │                   │            │
│       v                    v                   v            │
│  Retrieved +          Claude API          Markdown          │
│  User Context                             Documents          │
└─────────────────────────────────────────────────────────────┘
```

---

## Tier 1: Discovery

### Vector Database

**Platform**: Supabase with pgvector extension

**Tables**:
- `statistical_techniques` - 72+ fairness techniques with embeddings
- `documents` - Academic papers and course content
- `technique_implementations` - Python code for execution

### Retrieval Process

1. **Query Embedding**: Convert prompt to 1536-dimension vector
2. **Hybrid Search**: Combine vector similarity with metadata filters
3. **Ranking**: Score by relevance, complexity, and applicability
4. **Top-K Selection**: Return 5-10 most relevant techniques

### Metadata Filters

| Field | Purpose | Example Values |
|-------|---------|----------------|
| `intervention_stage` | When to apply | pre, in, post |
| `fairness_metric` | What it optimizes | demographic_parity, equal_opportunity |
| `complexity` | Implementation effort | low, medium, high |
| `domain` | Application context | lending, hiring, healthcare |

---

## Tier 2: Synthesis

### Context Assembly

Combine:
- Retrieved techniques from Tier 1
- User-provided constraints
- Domain-specific requirements
- Regulatory context

### LLM Synthesis

**Model**: Claude (Anthropic)

**Process**:
1. Structure context into prompt template
2. Generate consolidated deliverable
3. Validate output against schema
4. Extract citations and evidence chains

### Output Types

- Integration workflows
- Implementation guides
- Case study narratives
- Validation frameworks
- Executive presentations

---

## Example Query

**Input**:
```
Given a loan approval system with:
- Protected attributes: race, gender
- Fairness constraint: demographic parity ≥ 0.80
- Utility constraint: accuracy loss ≤ 5%
- Implementation time: < 1 week

Retrieve pre-processing techniques that:
1. Reduce correlation with protected attributes
2. Preserve predictive utility
3. Are production-ready
```

**Output**:
```
1. Disparate Impact Remover (Feldman et al., 2015)
   - Stage: pre-processing
   - Complexity: low
   - Expected DP improvement: 15-40%
   - Accuracy cost: 1-3%

2. Reweighting (Kamiran & Calders, 2012)
   - Stage: pre-processing
   - Complexity: low
   - Expected DP improvement: 10-25%
   - Accuracy cost: 0-2%
```

---

## Benefits

| Aspect | Traditional Approach | RAG-Based Approach |
|--------|---------------------|-------------------|
| Selection | Expert intuition | Evidence-based retrieval |
| Coverage | Limited to known techniques | 72+ indexed options |
| Consistency | Varies by practitioner | Systematic methodology |
| Documentation | Manual effort | Auto-generated citations |
| Reproducibility | Difficult | Fully reproducible |

---

## Navigation

- [Back to Methodology](./README.md)
- [Fairness Metrics](./fairness-metrics.md)
- [Validation Framework](./validation-framework.md)
