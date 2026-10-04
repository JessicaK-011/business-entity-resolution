# Business Entity Resolution with Multilingual Retrieval and Cross-Encoder Matching

A scalable machine learning pipeline for **business entity resolution** across noisy, heterogeneous data sources.

The task is to identify whether records from different sources refer to the **same real-world business**, even when names and addresses differ because of abbreviations, spelling variations, transliteration, missing components, punctuation, or inconsistent formatting.

Examples of challenging matches include:

```text
ABC Technologies Private Limited
12 MG Road, Bengaluru

↕ same entity?

ABC Tech Pvt Ltd
12 M.G. Rd, Bangalore
```

The pipeline is designed as a **multi-stage cascade**:

```text
Source 1 Record
      │
      ▼
Dense + Lexical Candidate Retrieval
      │
      ▼
Pairwise Feature Engineering
      │
      ▼
CatBoost Rejection Filter
      │
      ▼
Fine-tuned Cross-Encoder
      │
      ▼
Thresholded Match Predictions
```

---

## Problem Overview

Business information often comes from multiple independent sources and does not share a common identifier.

A single real-world entity may appear differently across datasets because of:

- spelling variations
- abbreviations such as `Pvt` vs `Private`
- legal suffix variations
- Unicode differences
- transliteration
- missing address components
- reordered address tokens
- punctuation changes
- typographical errors
- differences in municipal or postal formatting

The goal is therefore not simple exact matching, but learning whether two noisy records represent the same underlying entity.

---

## Approach

The solution uses a combination of:

- multilingual semantic embeddings
- lexical blocking
- engineered similarity features
- gradient-boosted filtering
- transformer-based pair classification

The overall objective is to maintain **high candidate recall** while keeping expensive neural inference computationally feasible.

---

## 1. Data Processing

Two separate representations of each business record are maintained.

### Original Text

The original business name and address are preserved for the neural models so that potentially useful information is not lost through aggressive normalization.

### Lexical Representation

A normalized version is created specifically for retrieval and feature engineering.

Processing includes operations such as:

- Unicode normalization
- whitespace normalization
- punctuation handling
- transliteration
- legal-suffix normalization
- selected business-token normalization
- common address abbreviation handling

Keeping the neural and lexical views separate allows the system to benefit from normalization without discarding information needed by the transformer model.

---

## 2. Candidate Generation

Performing pairwise classification against every record in the other sources would be prohibitively expensive.

The pipeline therefore begins with a high-recall candidate generation stage.

### Dense Retrieval

Business records are encoded using:

```text
intfloat/multilingual-e5-small
```

The encoder produces normalized **384-dimensional embeddings**.

Candidate similarity is computed using exact batched vector similarity.

For each Source 1 record, the system retrieves the most similar records from the other sources.

### Country Blocking

Retrieval is restricted to records sharing the supplied country label.

This significantly reduces the search space while preserving the assumption that genuine matches belong to the same country.

The country field is treated as an open categorical label rather than being hard-coded to particular countries.

### Lexical Candidate Routes

Dense retrieval is supplemented with lexical matching strategies so that candidates can still be recovered when embedding similarity alone is insufficient.

This is useful for cases involving:

- short company names
- strong numeric/address matches
- transliteration variants
- abbreviations
- rare tokens
- spelling differences

Multiple retrieval routes are merged into a single candidate pool.

### Candidate Cap

The final candidate set is capped to approximately **48 candidates per Source 1 record**.

This reduces the number of pairs that need to be evaluated by downstream models.

---

## 3. Pairwise Feature Engineering

For every retrieved candidate pair, the pipeline computes a set of **17 semantic, lexical, numeric, and retrieval-based features**.

These include:

### Semantic Features

- embedding cosine similarity

### Name Similarity Features

- normalized name edit similarity
- Unicode/raw name edit similarity
- token Jaccard similarity
- name length ratio
- missing-name indicator

### Address Similarity Features

- address edit similarity
- address token Jaccard similarity
- address length ratio
- missing-address indicator

### Numeric Features

- numeric-token Jaccard similarity
- missing-number indicator

Numeric agreement is particularly useful because addresses may look very similar semantically while containing different street or building numbers.

### Retrieval Features

- retrieval rank
- reciprocal/ranking score
- number of retrieval routes
- protected-candidate flag
- vendor/source indicator

These signals provide a compact tabular representation of how strongly two records appear related.

---

## 4. CatBoost Rejection Filter

A CatBoost classifier is trained on retrieved candidate pairs.

Its purpose is **not to make the final entity-matching decision**.

Instead, it acts as a conservative rejection filter that removes candidates that are very likely to be non-matches.

The cascade is therefore:

```text
Candidate Retrieval
        ↓
CatBoost Filter
        ↓
Cross-Encoder
```

rather than:

```text
Candidate Retrieval
        ↓
CatBoost Final Prediction
```

### Why use a rejection filter?

The cross-encoder is the most expensive stage of the system.

If millions of candidate pairs are passed through the transformer, inference becomes impractical.

The CatBoost model cheaply removes obvious negatives using the engineered features.

The filter threshold is selected conservatively so that candidate reduction does not significantly harm true-match recall.

Protected retrieval candidates bypass this rejection mechanism.

---

## 5. Cross-Encoder Matching

Candidates surviving the filtering stage are passed to a separately fine-tuned transformer cross-encoder.

Unlike dense retrieval, where the two records are independently embedded, the cross-encoder jointly reads both records.

Conceptually:

```text
[Reference Record]

Business Name:
ABC Technologies Private Limited

Address:
12 MG Road Bengaluru

[Candidate Record]

Business Name:
ABC Tech Pvt Ltd

Address:
12 M.G. Rd Bangalore
```

The model then predicts whether the pair refers to the same business.

This joint encoding allows the model to directly compare:

- spelling variants
- abbreviations
- token correspondences
- address components
- word ordering
- semantic relationships

The cross-encoder is therefore more accurate than the retrieval encoder, but also substantially more expensive.

This motivates the cascade architecture.

---

## 6. Hard-Negative Training

Training examples include retrieved candidates that are not true matches.

These are particularly useful because they often look deceptively similar to the reference record.

For example:

```text
Reference:
Sharma Medical Centre

Hard Negative:
Sharma Medical Services
```

Such examples teach the model to distinguish genuinely identical entities from superficially similar businesses.

Known positive pairs missed during candidate retrieval may be injected into the training set so the cross-encoder still learns from them.

This is restricted to training data only.

Validation and holdout evaluation use the actual retrieved candidates so that end-to-end performance remains realistic.

---

## 7. Validation Strategy

The labeled training data is divided into three groups:

```text
FIT
DEV
HOLDOUT
```

### FIT

Used to train:

- CatBoost
- cross-encoder

### DEV

Used for:

- rejection-threshold selection
- final match-threshold tuning
- cascade quality checks

### HOLDOUT

Used only to evaluate the final selected policy.

Splitting is performed at the **reference-entity level**, not individual candidate-pair level.

This prevents candidate pairs belonging to the same Source 1 entity from leaking across train and validation sets.

---

## 8. Evaluation

The system is optimized using **macro F0.5**.

```text
F0.5 = (1.25 × Precision × Recall) /
       (0.25 × Precision + Recall)
```

F0.5 gives greater importance to precision than recall.

This is appropriate for entity resolution because a false merge between two unrelated businesses is often more harmful than failing to link a true pair.

Threshold selection therefore favors conservative predictions.

Singleton entities are also important: predicting no match correctly is treated as a valid and meaningful outcome.

---

## 9. Unseen-Country Handling

The test data may contain countries not present in the labeled training data.

To avoid overfitting the pipeline to known countries:

- country values are discovered dynamically
- the model does not one-hot or hard-code only known training countries
- multilingual embeddings are used
- unvalidated countries can bypass the CatBoost rejection stage and proceed directly to the cross-encoder

This allows the system to generalize more safely to unseen geographical domains.

---

## 10. Scalability

The pipeline is designed to run within a constrained GPU notebook environment.

Several techniques are used to make the workflow feasible:

- exact batched embedding retrieval
- candidate caps
- lightweight tabular rejection filtering
- dual-GPU cross-encoder inference
- chunked processing
- resumable artifacts
- remote checkpointing

The goal is to spend expensive transformer inference only on candidate pairs that genuinely require deeper analysis.

---

## 11. Resumable Training and Inference

Long-running stages are checkpointed so interrupted notebook sessions can resume without restarting from zero.

Artifacts are stored in immutable parts.

Examples include:

- embedding chunks
- trained model checkpoints
- candidate parts
- prediction parts
- optimizer state
- epoch archives

Each stored artifact is accompanied by a SHA-256 digest to detect corruption or incompatible reuse.

Input fingerprints and helper-code hashes are also used to ensure that checkpoints from a different dataset or code version are not accidentally restored.

---

## 12. Multi-GPU Inference

The most expensive computation is cross-encoder scoring.

When two GPUs are available, independent minibatches are distributed across both devices.

The pipeline measures actual throughput instead of assuming ideal linear scaling.

This provides a more realistic estimate of total inference time.

---

## Pipeline Summary

```text
                       Business Record
                              │
                              ▼
              ┌───────────────────────────┐
              │ Text Normalization        │
              │ + Original Text Retained  │
              └───────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────┐
              │ Multilingual E5 Encoding  │
              └───────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────┐
              │ Dense + Lexical Retrieval │
              └───────────────────────────┘
                              │
                    ≤ ~48 candidates
                              │
                              ▼
              ┌───────────────────────────┐
              │ 17 Pairwise Features      │
              └───────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────┐
              │ CatBoost Rejection Filter │
              └───────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────┐
              │ Fine-Tuned Cross-Encoder  │
              └───────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────┐
              │ F0.5-Optimized Threshold  │
              └───────────────────────────┘
                              │
                              ▼
                      Final Entity Matches
```

---

## Tech Stack

### Machine Learning

- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- CatBoost
- Scikit-learn

### Data Processing

- Pandas
- NumPy

### NLP / Retrieval

- Multilingual E5 embeddings
- edit-distance similarity
- token Jaccard similarity
- lexical normalization
- transliteration

### Infrastructure

- Kaggle GPU environment
- dual-GPU inference
- Hugging Face private dataset storage
- resumable checkpointing

---

## Repository Structure

```text
business-entity-resolution/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebook/
│   └── kaggle_resumable_cascade.ipynb
│
├── src/
│   ├── retrieval.py
│   ├── features.py
│   ├── train_filter.py
│   ├── train_cross_encoder.py
│   └── inference.py
│
└── docs/
    └── methodology.md
```

---

## Running the Project

The complete pipeline is designed to run from the Kaggle notebook:

```text
notebook/kaggle_resumable_cascade.ipynb
```

The notebook executes the workflow sequentially:

```text
setup
  ↓
dataset validation
  ↓
train/dev/holdout split
  ↓
embedding generation
  ↓
candidate retrieval
  ↓
feature generation
  ↓
CatBoost training
  ↓
cross-encoder fine-tuning
  ↓
validation
  ↓
test retrieval
  ↓
test inference
  ↓
final output assembly
```

Long-running cells are resumable and completed remote artifacts are reused automatically.

---

## Outputs

The pipeline generates:

### `candidate_pairs.tsv`

Contains the retrieved candidate set considered by the matching pipeline.

### `matching_results.tsv`

Contains the final predicted Source 2 / Source 3 matches for every Source 1 record.

Final matches are always constrained to be a subset of the candidate set.

---

## Key Design Decisions

### Why not compare every possible pair?

The Cartesian product of the datasets would be far too large for transformer inference.

### Why use both dense and lexical retrieval?

Semantic embeddings handle meaning and spelling variation, while lexical signals recover candidates with strong token, number, or address overlap.

### Why use CatBoost before the cross-encoder?

It cheaply removes obvious negatives and reduces expensive transformer inference.

### Why not let CatBoost make the final prediction?

Handcrafted similarity features cannot capture all semantic relationships between noisy business records.

### Why use a cross-encoder?

Jointly encoding both records provides more precise pairwise comparison than independently generated embeddings.

### Why optimize F0.5?

False positive entity merges are more costly than missed matches, so precision is intentionally emphasized.

---

## Limitations

- Entity retrieval assumes true matches share the supplied country.
- Candidate generation places an upper bound on end-to-end recall.
- The CatBoost filter must be validated carefully before being used on unseen domains.
- Exact retrieval trades computational cost for better candidate recall.
- Large-scale execution requires significant GPU time and storage.

---

## Future Improvements

Potential extensions include:

- learned hybrid dense-sparse retrieval
- country-adaptive normalization
- stronger multilingual cross-encoders
- hard-negative mining across training epochs
- probability calibration
- uncertainty-aware thresholding
- approximate nearest-neighbor retrieval for larger datasets
- distillation of the cross-encoder into a faster reranker

---

## Notes

The competition dataset and trained checkpoints are intentionally **not included** in this repository.

No external business databases, geocoding APIs, or entity-resolution services are used.

The repository contains only the modeling and pipeline code required to reproduce the approach with the original challenge dataset.
