# Audit fixes — what changed and why

This file maps every finding in *AnonNet_Federated_KG_Completion_Comprehensive_Audit_Report.docx* to the change made in this archive,
and lists **additional defects found while fixing**. All notebooks, results CSVs, plots and the README were regenerated from the corrected
generators and re-executed from scratch (3 seeds each). Old results are **not** comparable with the new ones.

Environment of the final run: Python 3.12, torch 2.14.1 (CPU), torch_geometric 2.8.0, numpy 2.4, pandas 3.0 (see `run_metadata_*.json`).

## Headline: what the corrected results say

* The old LoadHalal "perfect 1.0000" scores were **invalid** (target-edge leakage **and** a tensor-orientation bug, N1 below).
* After the fixes the numbers are realistic, and **non-learned popularity/degree baselines (added) are strong**. On AnonNet the degree-product
  baseline beats every GNN on ROC-AUC/PR-AUC, so the GNNs are **not shown to learn more than degree structure**. On LoadHalal the RGCN is clearly
  better than the popularity baseline on ROC-AUC/PR-AUC but about equal on sampled-candidate MRR/Hits@10. See `README.md` §6 for the tables.
* Privacy: DP is **off**; the aggregation is an arithmetic **simulation** of pairwise masking. No privacy guarantee is claimed anywhere.

## Findings from the audit report

| ID | Status | Fix |
|---|---|---|
| **C1** LoadHalal target-edge leakage | **Fixed** | Target edges are split first. The message-passing graph is rebuilt from *explicit* edge sets (`build_msg_graph`): non-target relations + TRAIN target edges only (and, during training, only the share not currently used as supervision). Acceptance tests in `prepare()` assert that no held-out `has_manufacturer` edge appears in the forward or inverse relation of any message graph, and a deliberately leaky graph proves the test fires. |
| **C2** AnonNet reverse-edge leakage | **Fixed** | Unique *undirected* pairs (`u<v`) are split **before** reverse edges exist; reverse edges are added only for training pairs. Measured on the original protocol: **79.2%** of test rows had their reverse in the training graph. Tests assert no held-out pair in either direction in any message graph. |
| **C3** Structural features computed before the split | **Fixed** | Features are no longer precomputed. `make_x` computes log-degree (AnonNet) / per-relation log-degree (LoadHalal) from the **message-passing graph actually passed to the encoder** (training graph, a client's local graph, or the full training graph at evaluation). A test asserts held-out foods have zero target-relation degree. |
| **H1** Secure aggregation is a centralized simulation | **Fixed (claims) / partially (structure)** | Renamed and documented as *simulated pairwise masking* (`PairwiseMaskSimulation`). Masking now happens in a per-client function that uses only that client's pair secrets; the server function only sums masked payloads. Clients now send **model deltas** (not full state dicts). `run_privacy_simulation_tests()` checks cancellation and that a masked payload differs from the true update; it is labelled arithmetic-only. **Still not** a deployable protocol (no key agreement, authentication, dropout recovery). |
| **H2** DP not enabled / no accounting | **Fixed (claims) + mechanism added** | DP stays **off** (`DP_SIGMA = 0`) and every notebook says so. Added opt-in client-level DP-FedAvg (per-client clipping, uniform weights, noise calibrated to sensitivity `C/K`) and a Rényi accountant (`dp_epsilon`) with its assumptions documented. AnonNet §13 shows the utility/ε trade-off (large ε for usable utility = no meaningful guarantee with 4 clients). |
| **H3** Local negatives not client-restricted | **Fixed** | Clients build negatives from explicit pools of **their own** nodes (AnonNet) / own foods × own manufacturers (LoadHalal), in local ids. `assert_client_isolation` checks every endpoint of every local positive, negative and graph edge is owned by that client. |
| **H4** Global feature matrix visible to every client | **Fixed** | Each client receives only the static rows of nodes it owns (`STATIC_X[nodes_c]`), uses local ids, and computes degree features from its own local graph. Asserted in `assert_client_isolation`. |
| **H5** LoadHalal negative ownership | **Fixed** (same change as H3) | The sampler already excluded known positives; ownership is now enforced as well. |
| **H6** `manufacturer.csv` parsing fragile | **Fixed** | Explicit parser (split on the first comma) + validation: id format, uniqueness, row count, agreement with the naive parser. 815 names contain commas; the old `usecols` workaround would have truncated **816** names. |
| **M1** AnonNet is untyped | **Documented** | Notebook/README state it is homogeneous link prediction, not relation-aware KGC. |
| **M2** Cross-client edges discarded | **Reported + documented** | Cross-client shares are printed, saved in metadata and plotted. Federated training excludes them (stated everywhere). |
| **M3** Metric interpretation | **Fixed** | Threshold tuned on **validation only**; added Balanced Accuracy; class balance and majority-class accuracy reported; ties get average rank (a constant scorer can no longer look good). |
| **M4** Candidate pools | **Fixed (reporting)** | 50 filtered candidates (tail corruption for LoadHalal, either-endpoint for AnonNet), documented with random baselines; paper comparison relabelled *indicative, not like-for-like*. |
| **M5** Single run | **Fixed** | 3 seeds, mean ± std, per-seed CSVs. |
| **M6** Simulated clients | **Documented** | Stated as a single-process simulation everywhere. |
| **M7** Communication is an estimate | **Fixed (labelling) + measured serialisation** | Reports theoretical payload **and** measured pickle size of a client delta; explicitly *not* network traffic. |
| **D1/D2** README/comment overclaims | **Fixed** | README rewritten; notebook prose/comments rewritten to match the code. |
| **D3** Generators vs notebooks in sync | **Fixed** | Shared code lives in `notebook_common.py`; `check_notebooks_in_sync.py` regenerates into a temp dir and compares every cell (exit code 0 = in sync). |
| **D4** Saved CSVs not a fresh run | **Fixed** | Both notebooks re-executed end-to-end; CSVs/plots regenerated; `run_metadata_*.json` records versions, seeds, config, timestamp, protocol. |

## Additional defects found while fixing (not in the audit report)

| ID | Severity | Problem | Fix |
|---|---|---|---|
| **N1** | **Critical** | **LoadHalal edge tensors were `[E,2]` but the predictor indexed them as `[2,E]`** (`torch.LongTensor(edges).T` on an already `[2,E]` array). Every training step and every evaluation scored only **2 positive and 2 negative pairs**, which is why metrics were exactly 1.0000 and the ranking collapsed. Verified by a minimal reproduction. | All pair tensors are `[E,2]` end-to-end and `LinkPredictor` asserts `shape[1] == 2`. |
| **N2** | High | LoadHalal client ownership was derived from **all** `has_manufacturer` edges (including held-out ones) and, since each food has exactly one manufacturer, made every edge intra-client (0% cross-client) — contradicting the README's "~50% cross-client". | Ownership uses TRAIN edges only; held-out foods get a label-independent owner (`id mod K`). Train target edges are 100% intra-client *by construction* (stated), val/test target edges are ~75% cross-client (measured). |
| **N3** | High | Training-time message passing included the very edges used as supervision (predict edges you can see), creating a train/test mismatch. | Each epoch hides `SUP_FRAC` of train pairs from message passing and uses them as targets. |
| **N4** | Medium | Held-out foods could be sampled as always-negative training heads, biasing the model against them. | Training negatives are drawn only from train-positive heads (LoadHalal) / nodes seen in train pairs (AnonNet). |
| **N5** | Medium | Messages in the LoadHalal RGCN flowed only dst→src, so manufacturers never received information from their foods. | Inverse relations added (forward + reverse, standard RGCN), only for permitted edges. |
| **N6** | Medium | Centralized baseline trained 20 epochs vs 60 local epochs per federated client; no checkpoint selection. | Equal gradient steps; best checkpoint chosen on validation ROC-AUC for both. |
| **N7** | Medium | The "halal" use-case demo counted 43,760/43,101 foods (impossible), derived "halal manufacturers" from **all** edges including test labels, and silently skipped queries whose true manufacturer was filtered out. | Counts unique valid foods; manufacturer halal status from TRAIN edges only; excluded queries are counted and reported. |
| **N8** | Medium | Raw node-degree features were unscaled and also leaked held-out structure (C3). | Log-degree from the active graph only. |
| **N9** | Medium | Uniform random negatives make **popularity/degree** a very strong predictor, which the old notebooks never examined. | Added non-learned baselines (degree-product on AnonNet, manufacturer-popularity on LoadHalal) next to every table. |
| **N10** | Low | Data-quality issue previously only visible in logs: **98.3% of `food_ingredient` rows reference food ids missing from `food.csv`** (the `contains` relation has only 2,242 usable edges). | Reported with a prominent warning, plotted, saved in metadata. This is a source-data problem and cannot be fixed in code. |
| **N11** | Low | The AnonNet "node classification" cell generated synthetic labels but trained nothing, while README listed it as a feature. | Kept as an explicitly labelled stub; no result is claimed. |

## What was deliberately **not** changed / remaining limitations

* This is still a **single-process simulation**. There is no real Secure Aggregation protocol, no real network, no DP guarantee in the reported runs.
* Cross-client edges are excluded from federated training (a design choice, now measured and disclosed).
* Results are **not** comparable to the reference paper's tables (different splits, negatives, candidate pools).
* LoadHalal: the `contains` relation is nearly empty because of the source data; each food has exactly one manufacturer.
* The reference PDF and datasets are untouched.

## How to reproduce

```bash
pip install torch torch_geometric pandas numpy scikit-learn matplotlib nbconvert ipykernel
python generate_notebook.py && python generate_loadhalal.py        # regenerate notebooks
python check_notebooks_in_sync.py                                  # verify notebooks == generators
jupyter nbconvert --to notebook --execute --inplace anonnet_federated_kg_completion.ipynb
jupyter nbconvert --to notebook --execute --inplace loadhalal_federated_kg_completion.ipynb
```
Approximate runtime on one CPU core: AnonNet ≈ 3–4 min, LoadHalal ≈ 7 min (3 seeds + ablation + DP demo).
