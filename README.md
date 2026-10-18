# Simulated Federated Link Prediction for Supply-Chain Knowledge Graphs (research prototype)

Two notebooks study **centralized vs. simulated-federated** GNN link prediction on supply-chain graphs:

| Notebook | Data | Task | Models |
|---|---|---|---|
| `anonnet_federated_kg_completion.ipynb` | AnonNet (7,245 nodes, 41,063 undirected pairs, **no relation labels**) | homogeneous (untyped) link prediction | GraphSAGE, GAT |
| `loadhalal_federated_kg_completion.ipynb` | LoadHalal (68,444 nodes, 3 relations) | `has_manufacturer` (food → manufacturer) | RGCN |

> **Read this first.** This is a *single-process simulation*. It is **not** a validated privacy-preserving system.
> * **Differential privacy is OFF** in all reported runs (`DP_SIGMA = 0`). No (ε, δ) guarantee exists.
> * The aggregation is a **simulation of additive pairwise masking** (arithmetic cancels; one process; no key agreement,
>   authentication or dropout handling). It does **not** show that a real server cannot inspect individual updates.
> * Clients are partitions of one dataset in one process: no network, no access control.
> * Earlier versions of this repository reported perfect LoadHalal scores and made stronger privacy claims. Those were wrong; see `AUDIT_FIXES.md`.

## 1. Evaluation protocol (leakage-free)
* Target edges/pairs are **split first** (80/10/10). AnonNet splits *unique undirected pairs*, so a pair and its reverse always share a split.
* Held-out edges never enter any message-passing graph (either direction); structural features (log-degree) are computed **from the graph the encoder actually sees**.
  Mechanical acceptance tests run in `prepare()` for every seed.
* Each epoch hides 30% of train pairs from message passing and uses them as supervision targets (no "predict what you can see").
* Same number of gradient steps for centralized and federated; best checkpoint chosen on **validation ROC-AUC**; thresholded metrics use a threshold tuned on **validation only**.
* 1 positive : 5 negatives (majority-class accuracy = 0.833 → read Balanced Accuracy / ROC-AUC / PR-AUC). MRR / Hits@10 use **50 filtered candidates** per positive
  (random baseline MRR ≈ 0.089, Hits@10 = 0.196) and are **not comparable** with papers using other pools.
* 3 seeds (42, 43, 44), mean ± std. Non-learned baselines (degree product / manufacturer popularity) are reported next to the GNNs.

## 2. Federated setup
* **AnonNet:** 4 clients by `location` region. **LoadHalal:** 4 clients by manufacturer ownership (label-independent rule; see notebook §8).
* A client sees only its own nodes' static features, edges with both endpoints owned, negatives from its own nodes, and degree features from its own local graph.
  Cross-client edges are **excluded from federated training** (AnonNet: see notebook §9 for the share; LoadHalal: val/test target edges ≈75% cross-client).
* Clients send **model deltas**; the server sums (optionally masked) weighted deltas (FedAvg). Optional client-level DP-FedAvg + Rényi accountant exist (`DP_SIGMA > 0`), documented with assumptions in `notebook_common.py`.

## 3. Results (test set, mean ± std over 3 seeds)

**AnonNet** (`centralized_vs_federated.csv`)

| Model | Balanced Acc | ROC-AUC | PR-AUC | MRR | Hits@10 |
|---|---|---|---|---|---|
| GraphSAGE (central) | 0.834 ± 0.005 | 0.879 ± 0.005 | 0.573 ± 0.015 | 0.351 ± 0.025 | 0.744 ± 0.025 |
| GraphSAGE (federated) | 0.770 ± 0.017 | 0.839 ± 0.013 | 0.584 ± 0.037 | 0.323 ± 0.015 | 0.679 ± 0.019 |
| GAT (central) | 0.804 ± 0.009 | 0.880 ± 0.002 | 0.587 ± 0.005 | 0.401 ± 0.002 | 0.785 ± 0.003 |
| GAT (federated) | 0.784 ± 0.003 | 0.851 ± 0.005 | 0.599 ± 0.020 | 0.366 ± 0.007 | 0.737 ± 0.009 |
| *Degree-product baseline (non-learned)* | 0.817 ± 0.002 | **0.932 ± 0.000** | **0.774 ± 0.002** | 0.404 ± 0.008 | 0.732 ± 0.009 |

**LoadHalal** (`loadhalal_results.csv`)

| Model | Balanced Acc | ROC-AUC | PR-AUC | MRR | Hits@10 |
|---|---|---|---|---|---|
| RGCN (central) | 0.939 ± 0.006 | 0.982 ± 0.001 | 0.870 ± 0.010 | 0.444 ± 0.012 | 0.805 ± 0.020 |
| RGCN (federated) | 0.952 ± 0.005 | 0.983 ± 0.003 | 0.865 ± 0.026 | 0.466 ± 0.035 | 0.829 ± 0.037 |
| *Popularity baseline (non-learned)* | 0.784 ± 0.006 | 0.868 ± 0.003 | 0.630 ± 0.010 | 0.436 ± 0.008 | 0.786 ± 0.007 |

### How to read these numbers
* **AnonNet:** a non-learned degree baseline beats every GNN on ROC-AUC and PR-AUC (uniform random negatives favour low-degree nodes). The GNNs are **not shown to learn more than degree structure**.
  Federated training is a few points below centralized; this mixes the loss of cross-client edges with a local-vs-global degree-scale shift (not separated here).
* **LoadHalal:** the RGCN clearly beats the popularity baseline on ROC-AUC/PR-AUC/Balanced Accuracy but is **close to it on sampled-candidate MRR/Hits@10**. Removing identity features barely changes results
  (`loadhalal_ablation_id_features.csv`). In the harder all-manufacturer ranking (notebook §13) the true manufacturer's median rank is ≈125 of 2,520.
  The federated model matching the centralized one should not be over-interpreted: each food has exactly one manufacturer, so train target edges are 100% intra-client by construction.
* **DP illustration** (`dp_demo_results.csv`, AnonNet): ROC-AUC drops from 0.839 (no DP) to ≈0.74 at ε ≈ 13–31 (δ = 1e-5, client-level, 4 clients, no subsampling) — usable utility does not come with a meaningful guarantee here.
* **Reference paper:** numbers in the notebooks are labelled *indicative only*; splits, negatives and candidate pools differ.

## 4. Known limitations
* Source data: **98.3% of `food_ingredient` rows reference food ids absent from `food.csv`** (the `contains` relation keeps only 2,242 edges); 816 manufacturer names contain commas (now parsed correctly).
* Simulation only; no real Secure Aggregation, DP off, communication = measured serialised payload (not network traffic).
* AnonNet has no relation labels (not relation-aware KGC). The "node classification" cell only generates synthetic labels; no result is reported.

## 5. Repository layout
```
anonnet_federated_kg_completion.ipynb      executed notebook (outputs included)
loadhalal_federated_kg_completion.ipynb    executed notebook (outputs included)
generate_notebook.py / generate_loadhalal.py   generators (source of truth for the notebooks)
notebook_common.py                         shared code cells (metrics, sampling, aggregation simulation, DP accountant, tests)
check_notebooks_in_sync.py                 verifies notebooks == generators
AUDIT_FIXES.md                             audit finding -> fix map, plus additional defects found
centralized_vs_federated*.csv, federated_history.csv, loadhalal_*.csv, dp_demo_results.csv, run_metadata_*.json, plots/
AnonNet dataset/, Loadhalal dataset/, reference PDF
```

## 6. Reproduce
```bash
pip install torch torch_geometric pandas numpy scikit-learn matplotlib nbconvert ipykernel
python generate_notebook.py && python generate_loadhalal.py
python check_notebooks_in_sync.py
jupyter nbconvert --to notebook --execute --inplace anonnet_federated_kg_completion.ipynb
jupyter nbconvert --to notebook --execute --inplace loadhalal_federated_kg_completion.ipynb
```
≈3–4 min (AnonNet) and ≈7 min (LoadHalal) on one CPU core. Versions, seeds, config and timestamp are in `run_metadata_*.json`.

## 7. Reference
Peeris et al., "Advancing Sustainable Supply Chains Through Knowledge Graph Completion and Graph-Based Artificial Intelligence", *Sustainability* 2026, 18, 2825 (PDF included).
