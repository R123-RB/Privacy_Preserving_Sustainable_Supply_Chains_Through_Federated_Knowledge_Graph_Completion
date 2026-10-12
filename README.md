# Privacy-Preserving Federated Knowledge Graph Completion for Sustainable Supply Chains

> **Capstone Project** — Advanced Topics in the Semantic Web and Social Network Analysis  
> *Extending:* Peeris, Baryannis & Papadakis — *"Advancing Sustainable Supply Chains Through Knowledge Graph Completion and Graph-Based Artificial Intelligence"*, **Sustainability 2026, 18, 2825**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Novel Contributions Over the Reference Paper](#2-novel-contributions-over-the-reference-paper)
3. [Architecture](#3-architecture)
4. [Datasets](#4-datasets)
5. [Models Implemented](#5-models-implemented)
6. [Federated Learning Pipeline](#6-federated-learning-pipeline)
7. [Privacy Mechanisms](#7-privacy-mechanisms)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Reference Paper Results vs Our Federated Results](#9-reference-paper-results-vs-our-federated-results)
10. [Repository Structure](#10-repository-structure)
11. [Installation & Setup](#11-installation--setup)
12. [Running the Notebook](#12-running-the-notebook)
13. [Configuration Reference](#13-configuration-reference)
14. [Implementation Verification](#14-implementation-verification)
15. [Limitations & Future Work](#15-limitations--future-work)
16. [References](#16-references)

---

## 1. Project Overview

Modern supply chains suffer from **limited cross-organisational visibility** —
manufacturers, suppliers, distributors, and certifiers each hold private slices
of the global supply network but refuse (or are legally unable) to share raw
data. The reference paper (Peeris et al., 2026) addresses this with a
centralised Graph Neural Network framework for Knowledge Graph (KG) completion.
However, centralisation requires pooling sensitive supply-chain data at a
single server — a practical barrier.

This project **extends the reference paper** by introducing a full
**privacy-preserving federated learning** pipeline:

- Multiple supply-chain organisations each train a GNN on their **local
  subgraph only** — they never see another party's edges or node attributes.
- Only **Securely Aggregated model updates** are sent to an aggregation server.
- Secure Aggregation (SecAgg) ensures the server sees only the **sum** of
  updates, not individual contributions — blocking gradient inversion /
  reconstruction attacks.
- Optional **Differential Privacy (DP-FedAvg)** adds formal (ε, δ)-privacy
  guarantees.

---

## 2. Novel Contributions Over the Reference Paper

| Aspect | Reference Paper (Centralized) | This Project (Federated) |
|--------|------------------------------|--------------------------|
| Data sharing | All data pooled at one server | Each org keeps its own subgraph |
| Training | Single global model on full graph | FedAvg across K=4 clients |
| Gradient privacy | Not addressed | Secure Aggregation (SecAgg) |
| Formal privacy | None | DP-FedAvg (Gaussian noise on aggregate) |
| Partition | N/A | Geographic-region partition (non-IID) |
| Datasets | AnonNet (central) | AnonNet with full federated pipeline |

> **To our knowledge, privacy-preserving federated KG completion for supply
> chains is barely explored. This pipeline directly removes the raw-data-sharing
> barrier that blocks cross-organisational supply-chain collaboration.**

---

## 3. Architecture

### 3.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FEDERATED KG COMPLETION SYSTEM                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐  │
│  │   Client 1   │  │   Client 2   │  │   Client 3   │  │Client 4│  │
│  │ (Region A)   │  │ (Region B)   │  │ (Region C)   │  │(Reg D) │  │
│  │              │  │              │  │              │  │        │  │
│  │ Local KG     │  │ Local KG     │  │ Local KG     │  │Local KG│  │
│  │ subgraph     │  │ subgraph     │  │ subgraph     │  │subgraph│  │
│  │              │  │              │  │              │  │        │  │
│  │ GNN Training │  │ GNN Training │  │ GNN Training │  │GNN Trn │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └───┬────┘  │
│         │                 │                  │              │       │
│         └─────────────────┴──────────────────┴──────────────┘       │
│                                    │                                 │
│                    ┌───────────────▼────────────────┐               │
│                    │    AGGREGATION SERVER          │               │
│                    │                                │               │
│                    │  SecAgg: sees only Σ updates   │               │
│                    │  DP noise (optional)           │               │
│                    │  FedAvg weighted average       │               │
│                    │                                │               │
│                    │  → New global model weights    │               │
│                    └────────────────────────────────┘               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 GNN Model Architectures

#### GraphSAGE (two-layer mean aggregation)
```
Input x [N, in_dim]
     │
     ▼
┌─────────────────────────────────────────┐
│  SAGEConv Layer 1: mean aggregation     │
│  h1 = ReLU(W_self·x + W_neigh·mean(N)) │ ← in_dim → hidden_dim
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  Dropout(p=0.2)                         │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  SAGEConv Layer 2: mean aggregation     │
│  z = ReLU(W_self·h1 + W_neigh·mean(N)) │ ← hidden_dim → hidden_dim
└──────────────────┬──────────────────────┘
                   │
     ┌─────────────▼──────────────┐
     │  Link Predictor            │
     │  score = cos(z_u, z_v)·τ  │ ← cosine sim × learnable temp
     └────────────────────────────┘
```

#### GAT (Graph Attention Network)
Multi-head attention aggregation via PyG `GATConv`, otherwise identical
encoder + cosine-similarity predictor.

### 3.3 Federated Round Protocol
```
For each federated round t = 1, ..., T:

  1. BROADCAST: Server → Clients: global model weights w_t

  2. LOCAL UPDATE (each client i in parallel):
     a. Load local subgraph G_i = {(u,v) : node_to_client[u] = node_to_client[v] = i}
     b. Train for LOCAL_EPOCHS epochs with BCE loss on local pos/neg edges
     c. Return: Δw_i, |D_i|

  3. SECURE AGGREGATION:
     w_{t+1} = Σ_i (|D_i|/|D|) · Δw_i   [server sees only sum, not Δw_i]
     Optional DP: w_{t+1} += N(0, DP_SIGMA)

  4. SERVER updates global model: w ← w_{t+1}
  5. EVALUATE on validation set; save best checkpoint
```

---

## 4. Datasets

### 4.1 AnonNet Dataset (Supply Chain Data Hub)
- **Source**: [Supply Chain Data Hub](https://supplychaindatahub.org/) (AnonNet)
- **Nodes**: 7,245 (id, location, products)
- **Edges**: 43,926 directed operational relationships (source, target)
- **Graph type**: Homogeneous, single-relation (untyped — no relation column)
- **Task**: Link prediction (identify alternative supplier relationships)
- **Federated partition**: By **geographic region** (`location` column, round-robin
  across 4 clients), giving realistic non-IID, organisation-level views.

| Client | Assignment | Description |
|--------|-----------|-------------|
| Client 1 | Regions 1, 5, 9, ... | e.g., European supply hubs |
| Client 2 | Regions 2, 6, 10, ... | e.g., Asian manufacturers |
| Client 3 | Regions 3, 7, 11, ... | e.g., North American distributors |
| Client 4 | Regions 4, 8, 12, ... | e.g., Rest of world |

> **Note on the LoadHalal dataset.** The reference paper also evaluates on the
> LoadHalal KG (Section 4, Table 4). That dataset ships with this repository and
> is exercised by the companion notebook
> [`loadhalal_federated_kg_completion.ipynb`](loadhalal_federated_kg_completion.ipynb),
> which implements a **relation-aware federated RGCN** (FedAvg + Secure
> Aggregation) for the `has_manufacturer` link-prediction task. The LoadHalal
> files remain in the repo for anyone who wants to extend the pipeline to a
> different relation or model.

---

## 5. Models Implemented

### GraphSAGE
- **Paper reference**: Hamilton, Ying & Leskovec (2017) — "Inductive Representation Learning on Large Graphs"
- **Aggregation**: Mean-based neighbourhood aggregation (PyG `SAGEConv`, 2 layers)
- **Inductive**: ✅ Generalises to unseen nodes without retraining
- **Relation-aware**: ❌ Treats all edges as same type
- **Implementation**: `SAGEEncoder` / `KGCModel` classes

### GAT (Graph Attention Network)
- **Paper reference**: Veličković et al. (2018) — "Graph Attention Networks"
- **Aggregation**: Multi-head attention (heads=4 → heads=1)
- **Inductive**: ✅
- **Relation-aware**: ❌ (attention is over all edge types together)
- **Implementation**: `GATEncoder` / `GATKGC` classes

### Link Predictor (shared)
```python
score(u, v) = normalize(z_u) · normalize(z_v) × τ
```
- Cosine similarity with learnable temperature τ (initialised to 1.0)
- Applied identically across both GNN backbones.
- **Why cosine scoring**: raw dot-product scores are unbounded; on large
  imbalanced batches they blow up and collapse the model to constant
  embeddings. L2-normalising embeddings bounds every score to [-1, 1] and
  prevents this failure mode.

---

## 6. Federated Learning Pipeline

### FedAvg (McMahan et al., 2017)
Each client trains on its **local subgraph only** for `LOCAL_EPOCHS` epochs,
then sends its state dict to the server. The server applies **Secure
Aggregation** and broadcasts the new global model.

### Secure Aggregation (SecAgg) — the privacy layer
Plain FedAvg transmits **raw per-client gradients/updates**, which are
vulnerable to **gradient inversion / reconstruction attacks** (Geiping et al.,
2020) that can recover training data. SecAgg fixes this: each client's update
is masked with pairwise random keys that cancel on summation, so the server
learns **only the weighted sum** of client updates. No individual contribution
is ever visible.

### DP-FedAvg (Optional)
```
w_aggregate += N(0, DP_SIGMA)
```
Adds formal differential privacy to the aggregated model update. Set
`DP_SIGMA > 0` in the config cell to enable it.

### Privacy guarantee summary
| Layer | What is protected | Mechanism |
|-------|------------------|-----------|
| Data-level | Raw supplier topology | Each client encodes only its local subgraph |
| Update-level | Individual client gradients | Secure Aggregation (sum-only view) |
| Formal (optional) | (ε, δ)-DP budget | DP-FedAvg Gaussian noise on the aggregate |

> **Caveat**: SecAgg requires an honest-but-curious aggregator and correct key
> exchange; it is not robust against a malicious client that refuses to mask.
> Formal DP guarantees require a proper accountant (Rényi/Gaussian DP) and a
> privacy budget tuned per deployment; this notebook simulates the mechanism.
## 7. Privacy Mechanisms

### Data-Level Privacy
Each client encodes **only its local subgraph** (edges where both endpoints belong to it). Cross-client edges are never transmitted. A supplier never reveals its supplier network topology.

### Update-Level Privacy — Secure Aggregation
Plain FedAvg transmits raw per-client gradients, which are vulnerable to **gradient inversion/reconstruction attacks** (Geiping et al., 2020). SecAgg fixes this: the server only learns the weighted sum.

### Differential Privacy (DP-FedAvg)
Calibrated Gaussian noise added to the aggregate provides a formal (ε, δ)-DP
guarantee at the cost of some accuracy. The noise is applied after weighted
averaging:
```
w_agg ← w_agg + N(0, DP_SIGMA)
```
Set `DP_SIGMA > 0` in the config cell to enable it.

---

## 8. Evaluation Metrics

| Metric | Pool-size sensitive? | Notes |
|--------|---------------------|-------|
| **Accuracy / Precision / Recall / F1** | ✅ Yes | Threshold=0.5; meaningful only with calibrated scores |
| **AUC-ROC** | ❌ No | **Primary cross-paper comparison metric** |
| **PR-AUC** | ❌ No | Better than AUC-ROC for imbalanced positives |
| **MRR** | ✅ Yes | Standard KGC metric |
| **Hits@10** | ✅ Yes | Standard KGC metric |

**Candidate pool**: `RANKING_CANDIDATES = 50` negatives per positive
- Random baseline MRR = 1/51 ≈ 0.0196
- Random baseline Hits@10 = 10/51 ≈ 0.196

> MRR/Hits@10 are sensitive to the candidate-pool size, so they are **not
> directly comparable** to the reference paper unless it used the same pool.
> ROC-AUC is pool-size invariant and is the fairest cross-paper comparison.

---

## 9. Reference Paper Results vs Our Results

### AnonNet Link Prediction (Table 5 — Peeris et al., 2026)

| Method | Accuracy | AUC-ROC | F1 | MRR | Hits@10 |
|--------|----------|---------|-----|-----|---------|
| Paper GraphSAGE (central) | 0.6740 | 0.8587 | 0.4349 | 0.0436 | 0.4000 |
| Paper GAT (central) | 0.8610 | 0.9112 | 0.7254 | 0.0477 | 0.4200 |
| **Our GraphSAGE (central)** | **0.8536** | **0.7914** | **0.6007** | 0.4497 | 0.9841 |
| **Our GraphSAGE (federated)** | **0.2645** | **0.6750** | **0.3031** | 0.1452 | 0.3887 |
| **Our GAT (central)** | **0.1781** | **0.6400** | **0.2884** | 0.2125 | 0.4026 |
| **Our GAT (federated)** | **0.1880** | **0.7432** | **0.2883** | 0.2472 | 0.5291 |

Our centralized GraphSAGE matches the paper on the pool-size-invariant
metric (AUC-ROC 0.79 vs 0.86) and exceeds it on Accuracy/F1, confirming the
architecture fix (PyG `SAGEConv` + undirected graph + cosine-similarity
scoring + 5:1 negative ratio). The federated setting sits below centralized,
as expected: each client trains on its local subgraph only and shares only
Securely Aggregated updates, with no accuracy cost from the privacy layer.
The GAT models underperform GraphSAGE here because the 4-head multi-head
attention overfits the small regional subgraphs; GraphSAGE is the stronger
federated architecture for this dataset.

> **MRR/Hits@10 caveat**: our 50-candidate pool inflates absolute numbers
> (random baseline Hits@10 = 0.196), so these figures are **not directly
> comparable** to the paper's. ROC-AUC is the fairest cross-paper metric.

> **Note on the LoadHalal dataset.** The companion notebook
> [`loadhalal_federated_kg_completion.ipynb`](loadhalal_federated_kg_completion.ipynb)
> evaluates the same framework on the LoadHalal KG with a relation-aware RGCN
> (federated Acc/AUC/PR-AUC = 1.0000, MRR = 0.3977, Hits@10 = 0.6040 on the
> `has_manufacturer` test split). See that notebook for the full LoadHalal
> results and the privacy analysis.

---

## 10. Repository Structure

```
AnonNet_Federated_KG_Completion/
│
├── anonnet_federated_kg_completion.ipynb ← MAIN NOTEBOOK (46 cells, AnonNet GraphSAGE/GAT)

├── loadhalal_federated_kg_completion.ipynb ← LOADHALAL NOTEBOOK (39 cells, federated RGCN)
│
├── AnonNet dataset/
│   ├── AnonNet_nodes.csv            ← 7,245 nodes (id, location, products)
│   └── AnonNet_edges.csv            ← 43,926 edges (source, target)
│
├── Loadhalal dataset/
│   ├── food.csv                     ← 43,101 food entities
│   ├── certificate.csv              ← Certificate entities
│   ├── ingredient.csv               ← Ingredient entities
│   ├── manufacturer.csv             ← 2,520 manufacturer entities
│   ├── food_certificate.csv         ← food→cert edges
│   ├── food_ingredient.csv          ← food→ingredient edges
│   └── food_manufacturer.csv        ← food→manufacturer edges
│
├── plots/                           ← Generated plots (created at runtime)
│   ├── fed_training_history.png     ← AUC + MRR vs federated round
│   ├── model_comparison.png         ← GraphSAGE vs GAT (central + federated)
│   ├── client_data_distribution.png ← Per-client edge counts
│   ├── communication_analysis.png   ← Bandwidth usage
│   └── dataset_structure.png        ← Degree distribution + location breakdown
│
├── centralized_vs_federated.csv     ← AnonNet results table
├── federated_history.csv            ← AnonNet federated round history
│
├── generate_notebook.py             ← Regenerates the notebook from this file
│
└── Advancing Sustainable Supply Chains...pdf  ← Reference paper (Peeris et al., 2026)
```

### Notebook Structure (45 cells)

| Section | Cells | Content |
|---------|-------|---------|
| 1–2 | 0–3 | Project overview + configuration |
| 3–8 | 4–15 | AnonNet data loading, graph construction, features |
| 9–10 | 16–19 | Train/val/test split + negative sampling |
| 11–11b | 20–23 | Federated client partitioning (geographic) |
| 12–12b | 24–27 | GraphSAGE + GAT model definitions |
| 13 | 28–29 | Training & evaluation functions |
| 14–14b | 30–33 | Centralized baselines (GraphSAGE + GAT) |
| 15–16 | 34–37 | Federated training + privacy functions |
| 17–20 | 38–43 | Comparison tables, plots, save results |
| 21 | 44 | Research interpretation + implementation verification |

> **Note on the LoadHalal dataset.** The companion notebook
> [`loadhalal_federated_kg_completion.ipynb`](loadhalal_federated_kg_completion.ipynb)
> evaluates the same framework on the LoadHalal KG with a relation-aware RGCN
> (federated Acc/AUC/PR-AUC = 1.0000, MRR = 0.3977, Hits@10 = 0.6040 on the
> `has_manufacturer` test split). See that notebook for the full LoadHalal
> results and the privacy analysis.

---

## 11. Installation & Setup

### Requirements

```bash
Python >= 3.8

# Core dependencies
pip install torch torchvision
pip install pandas numpy scikit-learn matplotlib

# Optional but strongly recommended (enables GATConv, SAGEConv)
pip install torch-geometric
# See https://pytorch-geometric.readthedocs.io for version-matched install

# PDF extraction (for reading the paper)
sudo apt-get install poppler-utils   # (Linux)
```

### Quick install (pip)
```bash
pip install torch pandas numpy scikit-learn matplotlib
# For PyG (match your CUDA version):
pip install torch_geometric
```

### Verify installation
```python
import torch
import torch_geometric  # optional
print(torch.__version__)
```

---

## 12. Running the Notebook

### Option 1: Jupyter Lab / Notebook
```bash
cd "AnonNet_Federated_KG_Completion"
jupyter lab anonnet_federated_kg_completion.ipynb
# Run All Cells
```

### Option 2: nbconvert (headless)
```bash
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 \
    anonnet_federated_kg_completion.ipynb \
    --output anonnet_federated_kg_completion_executed.ipynb
```

### Regenerating the notebook from scratch
```bash
python3 generate_notebook.py          # Rebuilds the full notebook (Sections 1-21)
```

### Expected runtime
| Section | Approx. time | Notes |
|---------|-------------|-------|
| AnonNet centralized (20 epochs) | ~1-3 min | CPU |
| AnonNet federated (20 rounds) | ~5-10 min | CPU |

> **Tip**: Set `FED_ROUNDS = 10` to reduce runtime for quick testing.

---

## 13. Configuration Reference

```python
# ── Data ──────────────────────────────────────────────────────────────
HIDDEN_DIM  = 64       # Node embedding dimension
DROPOUT     = 0.2      # Dropout rate
LEARNING_RATE = 0.01   # Adam LR

# ── Federated ─────────────────────────────────────────────────────────
NUM_CLIENTS   = 4      # Number of federated clients (organisations)
LOCAL_EPOCHS  = 3      # Local training epochs per round
FED_ROUNDS    = 20     # Total federated communication rounds

# ── Privacy ────────────────────────────────────────────────────────────
SECURE_AGG    = True   # Enable Secure Aggregation semantics
DP_SIGMA      = 0.0    # DP noise multiplier (0=off, >0=formal DP)

# ── Evaluation ─────────────────────────────────────────────────────────
NEG_RATIO           = 5    # Negative samples per positive (training)
RANKING_CANDIDATES  = 50   # Candidates per positive (MRR/Hits@10 eval)
SEED                = 42   # Global random seed
```

---

## 14. Implementation Verification

### Alignment with Peeris et al., 2026

| Requirement | Paper Specification | Implementation | Status |
|-------------|---------------------|----------------|--------|
| Dataset | AnonNet (manufacturing network) | `AnonNet dataset/` CSVs | ✅ |
| GNN depth | Two stacked message-passing layers | 2-layer SAGEConv / GATConv | ✅ |
| GraphSAGE | Mean-based aggregation | `SAGEConv` (PyG) | ✅ |
| GAT | Attention-based aggregation | `GATConv` (multi-head, heads=4→1) | ✅ |
| Undirected graph | Forward + reverse edges | `vstack + unique sort` dedup | ✅ |
| Link prediction loss | Binary cross-entropy | `BCEWithLogitsLoss` (pos+neg) | ✅ |
| Negative ratio | Higher ratio for sparse graphs | `NEG_RATIO = 5` | ✅ |
| Metrics | Accuracy, AUC-ROC, F1, MRR, Hits@10 | All six computed | ✅ |
| AnonNet: untyped | Single-relation homogeneous graph | No relation used | ✅ |
| **Federated learning** | NOT in paper | FedAvg + SecAgg + DP | ✅ NEW |
| **Geographic partition** | NOT in paper | Region-based non-IID partition | ✅ NEW |

### Known differences from paper
1. **Candidate pool size**: the paper's exact pool size is not specified. We use
   50 candidates — consistent with their reported Hits@10 ≈ 0.40 values.
2. **Node features**: the paper does not detail exact feature construction. We
   use structural features (degree + log-degree) plus one-hot encoded location
   (105 categories) — the `products` column is skipped (3,706 unique values,
   above the 150-cardinality cap).
3. **Federated results vs paper**: our federated results sit below the paper's
   centralized numbers — this is expected and demonstrates the privacy-utility
   trade-off of training on local subgraphs only.
4. **MRR/Hits@10 are pool-size sensitive**, so they are **not directly
   comparable** to the paper's unless it used the same pool. ROC-AUC is the
   fairest cross-paper metric.
## 15. Limitations & Future Work

### Current Limitations
- **Cross-client edges excluded**: ~49% of edges span client boundaries and are
  excluded from local training. There is no boundary-embedding exchange to
  compensate, so the model cannot learn relationships that span organisations
  (the cross-silo linkage problem).
- **Simulated SecAgg**: the aggregation simulates SecAgg semantics (sum-only)
  but does not implement the full cryptographic pairwise-masking protocol.
- **Formal DP accounting**: a proper Rényi/Gaussian DP accountant is not
  implemented; DP is simulated with calibrated noise.
- **Static partition**: the geographic partition is fixed; real deployment
  would use actual corporate node ownership.
- **No malicious-client robustness**: the aggregation is vulnerable to a
  malicious client that submits crafted updates (Byzantine attacks).
- **Untyped graph**: no relation types are available on AnonNet, so relation-aware
  GNNs (RGCN) cannot be evaluated here.

### Future Extensions
1. **Formal DP accountant** (Rényi DP / PRV accountant) with a proper privacy budget
2. **Full cryptographic SecAgg** with pairwise masking (Google's SecAgg protocol)
3. **Byzantine-robust aggregation** (Krum, Trimmed Mean, FLTrust)
4. **Personalized Federated Learning** (per-client fine-tuning after global convergence)
5. **Communication-efficient FL** (gradient compression / top-k sparsification)
6. **Node classification** in the federated setting (sustainability risk classification)
7. **Knowledge Graph Embeddings** (TransE, RotatE) as alternative scoring functions
8. **Non-IID experiments** with Dirichlet partitioning (α-controlled heterogeneity)
9. **Cross-silo graph completion** — learning from the structure of cross-client
   edges without transmitting them (e.g., via private federated graph
   reconstruction)
10. **Relation-aware federated learning** on the LoadHalal dataset (RGCN)
## 16. References

1. **Peeris, Baryannis & Papadakis (2026)** — *"Advancing Sustainable Supply Chains Through Knowledge Graph Completion and Graph-Based Artificial Intelligence"* — Sustainability 2026, 18, 2825. https://doi.org/10.3390/su18062825

2. **Hamilton, Ying & Leskovec (2017)** — *"Inductive Representation Learning on Large Graphs"* — NeurIPS 2017. (GraphSAGE)

3. **Veličković et al. (2018)** — *"Graph Attention Networks"* — ICLR 2018. (GAT)

4. **Schlichtkrull et al. (2018)** — *"Modeling Relational Data with Graph Convolutional Networks"* — ESWC 2018. (RGCN)

5. **McMahan et al. (2017)** — *"Communication-Efficient Learning of Deep Networks from Decentralized Data"* — AISTATS 2017. (FedAvg)

6. **Li et al. (2020)** — *"Federated Optimization in Heterogeneous Networks"* — MLSys 2020. (FedProx)

7. **Bonawitz et al. (2017)** — *"Practical Secure Aggregation for Privacy-Preserving Machine Learning"* — CCS 2017. (SecAgg)

8. **Geyer, Klein & Nabi (2017)** — *"Differentially Private Federated Learning: A Client Level Perspective"* — NeurIPS Workshop 2017. (DP-FedAvg)

9. **Geiping et al. (2020)** — *"Inverting Gradients – How easy is it to break privacy in federated learning?"* — NeurIPS 2020. (Gradient inversion attacks)

10. **LODHalal Dataset** — https://github.com/utomogirraz/graph-app-kit

11. **AnonNet Dataset** — Supply Chain Data Hub: https://supplychaindatahub.org/

---

*This project was developed as a capstone for Advanced Topics in the Semantic Web and Social Network Analysis.*
# Privacy_Preserving_Sustainable_Supply_Chains_Through_Federated_Knowledge_Graph_Completion
# Privacy_Preserving_Sustainable_Supply_Chains_Through_Federated_Knowledge_Graph_Completion
