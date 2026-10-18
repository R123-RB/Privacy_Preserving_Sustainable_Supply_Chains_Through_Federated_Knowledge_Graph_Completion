"""Generates anonnet_federated_kg_completion.ipynb  (corrected version, see AUDIT_FIXES.md).

Run:  python generate_notebook.py
"""
import notebook_common as nc

nb = nc.Notebook()
add_md, add_code = nb.md, nb.code

# ---------------------------------------------------------------------------
add_md(r"""# 1. Project Overview
**Simulated federated link prediction on the AnonNet supply-chain graph** (research prototype).

Several simulated organisations each hold the part of the supply-chain graph that belongs to
them. Each trains a GNN locally; a server averages their model *deltas* (FedAvg).

**What this notebook is, precisely**
- A *single-process simulation* of cross-silo federated learning. Clients are partitions of one
  dataset, not separate machines, and there is no network, authentication or access control.
- AnonNet has **no relation labels**, so this is *homogeneous (untyped) link prediction* — a
  KGC-style baseline, **not** relation-aware multi-relational KGC. (The LoadHalal notebook is
  the multi-relational experiment.)
- The aggregation step is a **simulation of additive pairwise masking** (arithmetic only). It is
  *not* a deployable Secure Aggregation protocol.
- **Differential privacy is disabled** in the reported runs (`DP_SIGMA = 0`), so **no formal
  privacy guarantee is claimed.** A client-level DP-FedAvg mechanism and Renyi accountant are
  provided for experimentation (section 13).

**Evaluation protocol (leakage-free)** — undirected node pairs are split *before* reverse edges
exist; node-degree features and every message-passing graph are built from *training* pairs
only; validation/test pairs (and their reverses) never enter any message-passing graph.""")

add_code(nc.IMPORTS_CELL)

# ---------------------------------------------------------------------------
add_md("# 2. Configuration")
add_code(r'''# --- Data ---
NODE_ID_COL = None; SOURCE_COL = None; TARGET_COL = None      # None = auto-detect

def _find_dataset_dir():
    from pathlib import Path
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    candidates.append(Path.cwd())
    candidates.extend(Path.cwd().parents)
    for base in candidates:
        if (base / "AnonNet dataset" / "AnonNet_nodes.csv").exists():
            return base
    return Path(".")

_BASE_DIR = _find_dataset_dir()
NODES_PATH = str(_BASE_DIR / "AnonNet dataset" / "AnonNet_nodes.csv")
EDGES_PATH = str(_BASE_DIR / "AnonNet dataset" / "AnonNet_edges.csv")

# --- Model / training ---
HIDDEN_DIM = 64
DROPOUT = 0.2
LEARNING_RATE = 0.01
SUP_FRAC = 0.3            # per epoch, this share of TRAIN pairs is hidden from message passing and used as
                          # supervision targets (removes the train/test mismatch of "predict edges you can see")

# --- Federation ---
NUM_CLIENTS = 4
LOCAL_EPOCHS = 3
FED_ROUNDS = 20
CENTRAL_EPOCHS = FED_ROUNDS * LOCAL_EPOCHS      # same number of gradient steps as one federated client

# --- Evaluation protocol ---
NEG_RATIO = 5             # negatives per positive for training and for AUC / PR-AUC / thresholded metrics
RANKING_CANDIDATES = 50   # filtered candidates per positive for MRR / Hits@10 (random baseline MRR ~ 0.0875, Hits@10 ~ 0.196)
SPLIT = (0.8, 0.1, 0.1)
SEEDS = [42, 43, 44]      # every number in the results table is mean +/- std over these seeds

# --- Privacy-related switches (see section 6 for what they do / do not mean) ---
PAIRWISE_MASK_SIM = True  # SIMULATED pairwise masking of client deltas (arithmetic only)
DP_SIGMA = 0.0            # DP noise multiplier. 0.0 => differential privacy is OFF (no (eps, delta) guarantee)
DP_CLIP_NORM = 1.0        # per-client delta clipping norm (used only when DP_SIGMA > 0)
DP_DELTA = 1e-5
RUN_DP_DEMO = True        # small utility-vs-epsilon illustration in section 13

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEEDS[0])
print(f"seeds={SEEDS} | central epochs={CENTRAL_EPOCHS} | fed rounds={FED_ROUNDS}x{LOCAL_EPOCHS} local epochs | "
      f"DP {'ON sigma=%g' % DP_SIGMA if DP_SIGMA > 0 else 'OFF'} | pairwise-mask simulation={'ON' if PAIRWISE_MASK_SIM else 'OFF'}")''')

# ---------------------------------------------------------------------------
add_md("# 3. Load and Validate the Data")
add_code(r'''def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

nodes_df = load_csv(NODES_PATH)
edges_df = load_csv(EDGES_PATH)
print("nodes:", nodes_df.shape, nodes_df.columns.tolist())
print("edges:", edges_df.shape, edges_df.columns.tolist())
display(nodes_df.head(3)); display(edges_df.head(3))

def pick(df, candidates):
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None

NODE_ID_COL = NODE_ID_COL or pick(nodes_df, ['node_id', 'nodeid', 'id', 'entity_id', 'node'])
SOURCE_COL = SOURCE_COL or pick(edges_df, ['source', 'source_id', 'from', 'head', 'subject'])
TARGET_COL = TARGET_COL or pick(edges_df, ['target', 'target_id', 'to', 'tail', 'object'])
RELATION_COL = pick(edges_df, ['relation', 'relationship', 'predicate', 'edge_type', 'type', 'label'])
assert None not in (NODE_ID_COL, SOURCE_COL, TARGET_COL), "could not detect id/source/target columns"
print(f"columns: node id={NODE_ID_COL} source={SOURCE_COL} target={TARGET_COL} relation={RELATION_COL}")
print("=> no relation column: the graph is UNTYPED (homogeneous). This is link prediction, not relation-aware KGC."
      if RELATION_COL is None else f"relation column found: {RELATION_COL}")''')

# ---------------------------------------------------------------------------
add_md(r"""# 4. Graph Construction — unique *undirected* pairs

The edge list is turned into a set of **unique unordered node pairs** `(u, v)` with `u < v`.
The train/validation/test split is made on these pairs, so a pair and its reverse always land in
the same split (the previous version split after adding reverse edges, which left the reverse of
~80% of "test" edges in the training graph).""")
add_code(r'''nodes_df[NODE_ID_COL] = nodes_df[NODE_ID_COL].astype(str)
edges_df[SOURCE_COL] = edges_df[SOURCE_COL].astype(str)
edges_df[TARGET_COL] = edges_df[TARGET_COL].astype(str)

assert nodes_df[NODE_ID_COL].is_unique, "duplicate node ids"
node_list = nodes_df[NODE_ID_COL].tolist()
node2id = {n: i for i, n in enumerate(node_list)}
num_nodes = len(node_list)

total_raw = len(edges_df)
valid = edges_df[SOURCE_COL].isin(node2id) & edges_df[TARGET_COL].isin(node2id)
n_invalid = int((~valid).sum()); edges_valid = edges_df[valid]
loops = edges_valid[SOURCE_COL] == edges_valid[TARGET_COL]
n_loops = int(loops.sum()); edges_valid = edges_valid[~loops]
n_dup_dir = int(edges_valid.duplicated(subset=[SOURCE_COL, TARGET_COL]).sum())
edges_valid = edges_valid.drop_duplicates(subset=[SOURCE_COL, TARGET_COL])

s = edges_valid[SOURCE_COL].map(node2id).to_numpy(np.int64)
t = edges_valid[TARGET_COL].map(node2id).to_numpy(np.int64)
ALL_PAIRS = np.unique(np.stack([np.minimum(s, t), np.maximum(s, t)], 1), axis=0)     # [P, 2], u < v
n_reciprocal = len(edges_valid) - len(ALL_PAIRS)

print(f"nodes={num_nodes} | raw edges={total_raw} | invalid={n_invalid} | self-loops={n_loops} | "
      f"duplicate directed={n_dup_dir} | directed after cleaning={len(edges_valid)}")
print(f"reciprocal (u->v and v->u) collapsed into one undirected pair: {n_reciprocal}")
print(f"unique undirected pairs (the prediction targets): {len(ALL_PAIRS)}")
assert len(ALL_PAIRS) > 0
assert (ALL_PAIRS[:, 0] < ALL_PAIRS[:, 1]).all()

# Static, PUBLIC-per-owner node attributes only (no edge-derived information here).
def build_static_features(df, id_col, max_onehot=150):
    feats, names = [], []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == id_col: continue
        v = df[col].fillna(0).to_numpy(np.float32)
        feats.append(((v - v.mean()) / (v.std() + 1e-8)).reshape(-1, 1)); names.append(col)
    for col in df.select_dtypes(include=['object', 'category', 'string']).columns:
        if col == id_col: continue
        u = df[col].nunique()
        if u <= max_onehot:
            d = pd.get_dummies(df[col].fillna('__missing__'), prefix=col)
            feats.append(d.to_numpy(np.float32)); names += list(d.columns)
            print(f"one-hot '{col}' ({u} unique, missing -> '__missing__')")
        else:
            print(f"skipping '{col}': high cardinality ({u} unique values)")
    return torch.tensor(np.hstack(feats), dtype=torch.float32), names

STATIC_X, STATIC_NAMES = build_static_features(nodes_df, NODE_ID_COL)
IN_DIM = STATIC_X.shape[1] + 1       # + one structural feature (log-degree) computed from the *message-passing graph in use*
print(f"static feature matrix {tuple(STATIC_X.shape)}; model input dim = {IN_DIM} (log-degree is added per graph, never precomputed globally)")''')

# ---------------------------------------------------------------------------
add_md("# 5. Metrics, Sampling and Aggregation Utilities (shared with the LoadHalal notebook)")
add_code(nc.METRICS_CELL)
add_code(nc.SAMPLING_CELL)

add_md(r"""# 6. Aggregation: simulated pairwise masking + optional client-level DP
> **Scope warning.** The code below *simulates* additive pairwise masking inside one Python
> process and tests that the masks cancel. It does **not** implement a Secure Aggregation
> protocol (no key agreement, authentication, dropout recovery, or adversary model), so it does
> **not** show that a real server could not inspect an individual client's update. DP is **off**
> unless `DP_SIGMA > 0`, and even then the accountant's assumptions (documented in
> `dp_epsilon`) are narrow. What clients send are **model deltas**, weighted by the aggregation
> weight, not raw gradients.""")
add_code(nc.AGGREGATION_CELL)

# ---------------------------------------------------------------------------
add_md("# 7. Models")
add_code(r'''class SAGEEncoder(nn.Module):
    """Two-layer mean-aggregation GraphSAGE."""
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        if _HAS_PYG:
            self.conv1 = pyg_nn.SAGEConv(in_dim, hidden_dim)
            self.conv2 = pyg_nn.SAGEConv(hidden_dim, hidden_dim)
        else:   # minimal dense fallback
            self.fc1 = nn.Linear(2 * in_dim, hidden_dim)
            self.fc2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _mean_agg(x, ei):
        out = torch.zeros_like(x); cnt = torch.zeros(x.size(0), 1)
        if ei.numel():
            out.index_add_(0, ei[1], x[ei[0]]); cnt.index_add_(0, ei[1], torch.ones(ei.size(1), 1))
        return out / cnt.clamp(min=1)

    def forward(self, x, ei):
        if _HAS_PYG:
            h = self.dropout(F.relu(self.conv1(x, ei)))
            return F.relu(self.conv2(h, ei))
        h = self.dropout(F.relu(self.fc1(torch.cat([x, self._mean_agg(x, ei)], 1))))
        return F.relu(self.fc2(torch.cat([h, self._mean_agg(h, ei)], 1)))

class GATEncoder(nn.Module):
    """Two-layer GAT (PyG GATConv, 4 heads in layer 1)."""
    def __init__(self, in_dim, hidden_dim, dropout, heads=4):
        super().__init__()
        self.conv1 = pyg_nn.GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = pyg_nn.GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, ei):
        h = self.dropout(F.elu(self.conv1(x, ei)))
        return F.elu(self.conv2(h, ei))

class LinkPredictor(nn.Module):
    """Cosine similarity x learnable temperature. Input: pairs LongTensor [E, 2]."""
    def __init__(self, init_tau=1.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(init_tau)))

    def forward(self, z, pairs):
        assert pairs.dim() == 2 and pairs.size(1) == 2, f"pairs must be [E, 2], got {tuple(pairs.shape)}"
        z = F.normalize(z, p=2, dim=-1)
        return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(-1) * self.tau

class KGCModel(nn.Module):
    def __init__(self, kind, in_dim, hidden_dim, dropout):
        super().__init__()
        self.kind = kind
        self.encoder = SAGEEncoder(in_dim, hidden_dim, dropout) if kind == 'GraphSAGE' else GATEncoder(in_dim, hidden_dim, dropout)
        self.predictor = LinkPredictor()

KINDS = ['GraphSAGE'] + (['GAT'] if _HAS_PYG else [])
def make_model(kind):
    return KGCModel(kind, IN_DIM, HIDDEN_DIM, DROPOUT)
print("models:", KINDS, "| parameters:", {k: sum(p.numel() for p in make_model(k).parameters()) for k in KINDS})''')

# ---------------------------------------------------------------------------
add_md(r"""# 8. Graph helpers, training step, evaluation
`make_x` computes the structural feature (log-degree) **from the message-passing graph that is
actually passed to the encoder** — the training graph (with supervision pairs hidden), a client's
local graph, or the full training graph at evaluation. Held-out pairs can therefore never
influence a feature.""")
add_code(r'''def bidir(pairs):
    """LongTensor [E, 2] of undirected pairs -> edge_index [2, 2E] with both directions."""
    return torch.cat([pairs.T, pairs.flip(1).T], dim=1)

def make_x(static_x, edge_index, num_nodes):
    deg = torch.bincount(edge_index[0], minlength=num_nodes).float() if edge_index.numel() else torch.zeros(num_nodes)
    return torch.cat([static_x, torch.log1p(deg).unsqueeze(1)], dim=1)

def train_step(model, opt, static_x, num_nodes, pairs, neg_pool, gen):
    """One full-batch step. A random SUP_FRAC of `pairs` is hidden from message passing and used as
    positive supervision; the rest forms the message-passing graph (+ reverse directions)."""
    E = pairs.size(0)
    if E < 2:
        return float('nan')
    model.train(); opt.zero_grad()
    perm = torch.randperm(E, generator=gen)
    n_sup = max(1, int(round(SUP_FRAC * E)))
    sup, msg = pairs[perm[:n_sup]], pairs[perm[n_sup:]]
    ei = bidir(msg)
    z = model.encoder(make_x(static_x, ei, num_nodes), ei)
    neg = neg_pool[torch.randint(0, neg_pool.size(0), (NEG_RATIO * n_sup,), generator=gen)]
    ps, ns = model.predictor(z, sup), model.predictor(z, neg)
    loss = F.binary_cross_entropy_with_logits(ps, torch.ones_like(ps)) + \
           F.binary_cross_entropy_with_logits(ns, torch.zeros_like(ns))
    loss.backward(); opt.step()
    return float(loss.detach())

@torch.no_grad()
def evaluate_model(model, ctx, split, thr=None):
    """Scores `split` ('val' | 'test') on the FULL TRAINING graph. The threshold for the thresholded
    metrics is tuned on validation scores and reused unchanged for test."""
    model.eval()
    ei = bidir(ctx['train_pairs'])
    z = F.normalize(model.encoder(make_x(ctx['static_x'], ei, ctx['num_nodes']), ei), dim=-1)
    tau = model.predictor.tau
    sc = lambda p: ((z[p[:, 0]] * z[p[:, 1]]).sum(-1) * tau).cpu().numpy()
    b = ctx[split]
    pos, neg = sc(b['pos']), sc(b['neg'])
    P, K = b['cand'].shape[:2]
    cand = sc(b['cand'].reshape(-1, 2)).reshape(P, K)
    thr = best_f1_threshold(pos, neg) if thr is None else thr
    return summarize_scores(pos, neg, pos, cand, thr)

def degree_product_baseline(ctx):
    """NON-LEARNED reference (preferential attachment): score(u, v) = log(1+deg_u) + log(1+deg_v) on the TRAIN graph.
    Uniform random negatives favour low-degree nodes, so degree alone can score well; the GNN must beat this."""
    deg = np.bincount(ctx['train_pairs'].numpy().reshape(-1), minlength=ctx['num_nodes'])
    sc = lambda p: (np.log1p(deg[p[:, 0].numpy()]) + np.log1p(deg[p[:, 1].numpy()])).astype(np.float64)
    def run(split, thr=None):
        b = ctx[split]; pos, neg = sc(b['pos']), sc(b['neg']); P, K = b['cand'].shape[:2]
        cand = sc(b['cand'].reshape(-1, 2)).reshape(P, K)
        return summarize_scores(pos, neg, pos, cand, best_f1_threshold(pos, neg) if thr is None else thr)
    return run('test', run('val')['Threshold'])''')

# ---------------------------------------------------------------------------
add_md(r"""# 9. Federated Partition (by geographic region) and Per-Experiment Preparation
Ownership is defined by the node attribute `location` (not by edges), so it cannot encode held-out
edge information. A client sees **only**: its own nodes' static attributes, edges whose *both*
endpoints it owns, negatives drawn from *its own* node set, and degree features computed from its
own local graph. `assert_client_isolation` checks this mechanically.""")
add_code(r'''node_locations = nodes_df['location'].fillna('Unknown').astype(str).to_numpy()
unique_locations = sorted(set(node_locations))
location_to_client = {loc: i % NUM_CLIENTS for i, loc in enumerate(unique_locations)}
NODE_TO_CLIENT = np.array([location_to_client[l] for l in node_locations])
for c in range(NUM_CLIENTS):
    print(f"Client {c+1}: {(NODE_TO_CLIENT == c).sum()} nodes in {sum(1 for l in unique_locations if location_to_client[l] == c)} regions")

def assert_no_heldout_in_graph(edge_index, heldout_pairs, name):
    """No held-out pair, in EITHER direction, may exist in a message-passing graph."""
    ei = edge_index.numpy().T
    g = make_key_array(ei, num_nodes, undirected=True)
    h = make_key_array(heldout_pairs, num_nodes, undirected=True)
    assert not np.isin(h, g).any(), f"LEAK: held-out pair present in {name}"

def assert_client_isolation(client, expected_local_edges):
    n = client['num_nodes']
    assert client['static_x'].shape[0] == n, "client holds features for nodes it does not own"
    for name in ('pairs', 'neg_pool'):
        t = client[name]
        assert t.ndim == 2 and t.shape[1] == 2, f"{name} must be [E, 2]"
        assert t.numel() == 0 or (int(t.min()) >= 0 and int(t.max()) < n), f"{name} references a node outside the client"
        owners = NODE_TO_CLIENT[client['nodes_global'][t.numpy().reshape(-1)]]
        assert (owners == client['client_id']).all(), f"{name} contains an endpoint owned by another client"
    assert client['pairs'].size(0) == expected_local_edges

def build_clients(ctx, seed):
    tr = ctx['train_pairs'].numpy()
    clients = []
    for c in range(NUM_CLIENTS):
        nodes_c = np.where(NODE_TO_CLIENT == c)[0]
        g2l = -np.ones(num_nodes, dtype=np.int64); g2l[nodes_c] = np.arange(len(nodes_c))
        mask = (NODE_TO_CLIENT[tr[:, 0]] == c) & (NODE_TO_CLIENT[tr[:, 1]] == c)
        local_pairs = g2l[tr[mask]]                                        # LOCAL ids
        rng = np.random.default_rng(seed * 100 + 10 + c)
        forb = make_key_array(local_pairs, len(nodes_c), True)
        pool_n = max(NEG_RATIO * len(local_pairs) * 2, 16)
        neg = sample_negative_pairs(pool_n, rng, np.arange(len(nodes_c)), np.arange(len(nodes_c)), forb, len(nodes_c), True)
        client = {'client_id': c, 'nodes_global': nodes_c, 'num_nodes': len(nodes_c),
                  'static_x': STATIC_X[nodes_c].clone(),                    # ONLY this client's rows
                  'pairs': torch.LongTensor(local_pairs), 'neg_pool': torch.LongTensor(neg)}
        assert_client_isolation(client, int(mask.sum()))
        clients.append(client)
    return clients

def prepare(seed):
    """Everything for one seed: split -> negatives -> ranking candidates -> clients, with leakage assertions."""
    rng = np.random.default_rng(seed)
    P = len(ALL_PAIRS); perm = rng.permutation(P)
    n_tr, n_va = int(SPLIT[0] * P), int(SPLIT[1] * P)
    train, val, test = ALL_PAIRS[perm[:n_tr]], ALL_PAIRS[perm[n_tr:n_tr + n_va]], ALL_PAIRS[perm[n_tr + n_va:]]

    k_all = make_key_array(ALL_PAIRS, num_nodes, True)
    k_tr, k_va, k_te = (make_key_array(a, num_nodes, True) for a in (train, val, test))
    assert not (np.isin(k_tr, k_va).any() or np.isin(k_tr, k_te).any() or np.isin(k_va, k_te).any()), "splits overlap"
    assert len(k_tr) + len(k_va) + len(k_te) == len(k_all), "split does not partition the pair set"

    allnodes = np.arange(num_nodes)
    # training negatives avoid TRAIN positives and use only nodes seen in training; evaluation negatives avoid ALL known positives
    train_nodes = np.unique(train.reshape(-1))      # nodes seen in TRAIN pairs only
    train_neg = sample_negative_pairs(NEG_RATIO * len(train) * 2, rng, train_nodes, train_nodes, k_tr, num_nodes, True)
    def bundle(pairs):
        neg = sample_negative_pairs(NEG_RATIO * len(pairs), rng, allnodes, allnodes, k_all, num_nodes, True)
        cand = build_ranking_candidates(pairs, RANKING_CANDIDATES, rng, allnodes, k_all, num_nodes, True, 'either')
        assert not in_keys(k_all, make_key_array(neg, num_nodes, True)).any()
        return {'pos': torch.LongTensor(pairs), 'neg': torch.LongTensor(neg), 'cand': torch.LongTensor(cand)}
    ctx = {'seed': seed, 'num_nodes': num_nodes, 'static_x': STATIC_X, 'train_pairs': torch.LongTensor(train),
           'train_neg_pool': torch.LongTensor(train_neg), 'val': bundle(val), 'test': bundle(test)}

    # ---- acceptance tests: no held-out pair (or its reverse) in ANY message-passing graph ----
    heldout = np.concatenate([val, test])
    assert_no_heldout_in_graph(bidir(ctx['train_pairs']), heldout, "the evaluation message-passing graph")
    sup_perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    assert_no_heldout_in_graph(bidir(ctx['train_pairs'][sup_perm[int(SUP_FRAC * len(train)):]]), heldout, "a training message-passing graph")
    # sensitivity check: the test WOULD fire if a held-out pair's reverse were present
    leaky = torch.cat([ctx['train_pairs'], torch.LongTensor(test[:1]).flip(1)])
    fired = False
    try:
        assert_no_heldout_in_graph(bidir(leaky), heldout, "deliberately leaky graph")
    except AssertionError:
        fired = True
    assert fired, "leak test is insensitive"

    ctx['clients'] = build_clients(ctx, seed)
    c2c = lambda a: float((NODE_TO_CLIENT[a[:, 0]] != NODE_TO_CLIENT[a[:, 1]]).mean())
    ctx['stats'] = {'train_pairs': len(train), 'val_pairs': len(val), 'test_pairs': len(test),
                    'cross_client_train': c2c(train), 'cross_client_val': c2c(val), 'cross_client_test': c2c(test),
                    'local_train_edges_per_client': [int(c['pairs'].size(0)) for c in ctx['clients']]}
    return ctx

ctx0 = prepare(SEEDS[0])
st = ctx0['stats']
print(f"split (seed {SEEDS[0]}): train={st['train_pairs']} val={st['val_pairs']} test={st['test_pairs']} undirected pairs")
print(f"cross-client share  train={st['cross_client_train']:.1%}  val={st['cross_client_val']:.1%}  test={st['cross_client_test']:.1%}")
print(f"=> clients train only on the {1 - st['cross_client_train']:.1%} intra-client pairs; cross-client pairs are EXCLUDED from federated training "
      f"(the federated model never learns from them), yet they are part of the evaluation.")
print("local training pairs per client:", st['local_train_edges_per_client'])
print("leakage acceptance tests passed: disjoint splits, no held-out pair or reverse in any message graph, client isolation OK")''')

# ---------------------------------------------------------------------------
add_md(r"""# 10. Centralized and Federated Training
Both settings use the same number of gradient steps per client (`CENTRAL_EPOCHS = FED_ROUNDS x LOCAL_EPOCHS`),
select the best checkpoint on **validation ROC-AUC**, and report test metrics with a threshold tuned on
validation only. The federated server scores validation/test pairs on the global training graph — this is an
*evaluation harness*; in a real deployment no single party would hold that graph.""")
add_code(r'''def fit_central(kind, ctx, seed):
    set_seed(seed)
    model = make_model(kind); opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    gen = torch.Generator().manual_seed(seed)
    best, hist = {'auc': -1.0}, []
    for ep in range(1, CENTRAL_EPOCHS + 1):
        loss = train_step(model, opt, ctx['static_x'], ctx['num_nodes'], ctx['train_pairs'], ctx['train_neg_pool'], gen)
        if ep % LOCAL_EPOCHS == 0:
            m = evaluate_model(model, ctx, 'val')
            hist.append({'step': ep // LOCAL_EPOCHS, 'loss': loss, 'val_auc': m['ROC-AUC'], 'val_mrr': m['MRR']})
            if m['ROC-AUC'] > best['auc']:
                best = {'auc': m['ROC-AUC'], 'state': copy.deepcopy(model.state_dict()), 'step': ep // LOCAL_EPOCHS}
    model.load_state_dict(best['state'])
    thr = evaluate_model(model, ctx, 'val')['Threshold']
    return {'test': evaluate_model(model, ctx, 'test', thr), 'val_auc': best['auc'], 'best_step': best['step'], 'history': hist}

def client_update(kind, global_state, client, seed, rnd):
    """Runs on the client: sees only its own static features / local graph / local negatives."""
    model = make_model(kind); model.load_state_dict(global_state)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    gen = torch.Generator().manual_seed(seed * 100003 + rnd * 101 + client['client_id'])
    n = client['pairs'].size(0)
    if n >= 2:
        for _ in range(LOCAL_EPOCHS):
            train_step(model, opt, client['static_x'], client['num_nodes'], client['pairs'], client['neg_pool'], gen)
    else:
        n = 0
    new = model.state_dict()
    delta = {k: (new[k] - global_state[k]).float() for k in global_state}     # model DELTA, not raw gradients
    return delta, n

def fit_federated(kind, ctx, seed, dp_sigma=None, clip_norm=None):
    set_seed(seed)
    global_model = make_model(kind)
    masker = PairwiseMaskSimulation(NUM_CLIENTS, seed)
    best, hist, uplink = {'auc': -1.0}, [], []
    for rnd in range(1, FED_ROUNDS + 1):
        gstate = {k: v.clone() for k, v in global_model.state_dict().items()}
        results = [client_update(kind, gstate, c, seed, rnd) for c in ctx['clients']]
        agg, info = aggregate_round(results, rnd, masker, dp_sigma=dp_sigma, clip_norm=clip_norm)
        if agg is not None:
            global_model.load_state_dict({k: gstate[k] + agg[k] for k in gstate})
            uplink.append(info['uplink_bytes_per_client'])
        m = evaluate_model(global_model, ctx, 'val')
        hist.append({'step': rnd, 'val_auc': m['ROC-AUC'], 'val_mrr': m['MRR'], 'val_f1': m['F1-score'], 'val_prauc': m['PR-AUC'], 'val_hits10': m['Hits@10']})
        if m['ROC-AUC'] > best['auc']:
            best = {'auc': m['ROC-AUC'], 'state': copy.deepcopy(global_model.state_dict()), 'step': rnd}
    global_model.load_state_dict(best['state'])
    thr = evaluate_model(global_model, ctx, 'val')['Threshold']
    return {'test': evaluate_model(global_model, ctx, 'test', thr), 'val_auc': best['auc'], 'best_step': best['step'],
            'history': hist, 'uplink_bytes_per_client_round': float(np.mean(uplink)) if uplink else 0.0,
            'num_params': sum(p.numel() for p in global_model.parameters())}
print("training functions defined")''')

add_code(r'''BASE = 'Degree-product baseline (non-learned)'
RESULTS = {f'{k} ({s})': [] for k in KINDS for s in ('Centralized', 'Federated')}
RESULTS[BASE] = []
HISTORY, COMM, STATS_PER_SEED, RUN_LOG = {}, {}, {}, []
t_all = time.time()
for seed in SEEDS:
    ctx = ctx0 if seed == SEEDS[0] else prepare(seed)
    STATS_PER_SEED[seed] = ctx['stats']
    RESULTS[BASE].append(degree_product_baseline(ctx))
    print(f"seed {seed} | degree-product baseline (non-learned): AUC={RESULTS[BASE][-1]['ROC-AUC']:.4f} MRR={RESULTS[BASE][-1]['MRR']:.4f}")
    for kind in KINDS:
        t0 = time.time()
        cen = fit_central(kind, ctx, seed)
        fed = fit_federated(kind, ctx, seed)
        RESULTS[f'{kind} (Centralized)'].append(cen['test']); RESULTS[f'{kind} (Federated)'].append(fed['test'])
        if seed == SEEDS[0]:
            HISTORY[kind] = {'central': cen['history'], 'federated': fed['history']}
        COMM.setdefault(kind, []).append((fed['num_params'], fed['uplink_bytes_per_client_round']))
        RUN_LOG.append({'seed': seed, 'model': kind, 'central_best_step': cen['best_step'], 'fed_best_round': fed['best_step']})
        print(f"seed {seed} | {kind:9s} | central AUC={cen['test']['ROC-AUC']:.4f} MRR={cen['test']['MRR']:.4f} | "
              f"federated AUC={fed['test']['ROC-AUC']:.4f} MRR={fed['test']['MRR']:.4f} | {time.time() - t0:.0f}s")
print(f"total time {time.time() - t_all:.0f}s")''')

# ---------------------------------------------------------------------------
add_md("# 11. Results (mean ± std over seeds)")
add_code(r'''summary = mean_std_table(RESULTS)
per_seed_rows = []
for col, runs in RESULTS.items():
    for seed, r in zip(SEEDS, runs):
        per_seed_rows.append({'model_setting': col, 'seed': seed, **r})
pd.DataFrame(per_seed_rows).to_csv('centralized_vs_federated_per_seed.csv', index=False)
summary.to_csv('centralized_vs_federated.csv', index=False)

pd.set_option('display.float_format', lambda x: f'{x:.4f}')
display(summary)
pool_note = (f"Protocol: {len(SEEDS)} seeds; validation-tuned F1 threshold; class balance 1 positive : {NEG_RATIO} negatives "
             f"(majority-class accuracy = {NEG_RATIO / (1 + NEG_RATIO):.3f}); ranking = filtered either-endpoint corruption with "
             f"{RANKING_CANDIDATES} candidates per positive (random baseline MRR ~ {np.mean(1 / np.arange(1, RANKING_CANDIDATES + 2)):.4f}, "
             f"Hits@10 = {10 / (RANKING_CANDIDATES + 1):.4f}).")
print(pool_note)

hist_df = pd.DataFrame(HISTORY['GraphSAGE']['federated']).rename(columns={'step': 'round'})
hist_df.rename(columns={'val_auc': 'auc', 'val_f1': 'f1', 'val_prauc': 'prauc', 'val_mrr': 'mrr', 'val_hits10': 'hits10'}).to_csv('federated_history.csv', index=False)
print("saved centralized_vs_federated.csv, centralized_vs_federated_per_seed.csv, federated_history.csv")''')

add_md(r"""## Reference-paper numbers (indicative only — **not** a like-for-like comparison)
The reference paper (Peeris et al., *Sustainability* 2026, 18, 2825, Table 5) reports centralized
GraphSAGE/GAT on AnonNet. Its candidate pool, negative-sampling rule and split are not matched
here, so thresholded metrics, MRR and Hits@10 are **not directly comparable**. ROC-AUC is the least
protocol-sensitive number, but differences in splitting still apply.""")
add_code(r'''paper = pd.DataFrame({
    'Method': ['Paper GraphSAGE', 'Paper GAT'],
    'Accuracy': [0.6740, 0.8610], 'ROC-AUC': [0.8587, 0.9112], 'F1': [0.4349, 0.7254],
    'MRR': [0.0436, 0.0477], 'Hits@10': [0.4000, 0.4200]})
display(paper)
ours = summary.set_index('Metric')
rows = []
for k in KINDS:
    rows.append({'Method': f'This work {k} (centralized)', 'Accuracy': ours.loc['Accuracy', f'{k} (Centralized) mean'],
                 'ROC-AUC': ours.loc['ROC-AUC', f'{k} (Centralized) mean'], 'F1': ours.loc['F1-score', f'{k} (Centralized) mean'],
                 'MRR': ours.loc['MRR', f'{k} (Centralized) mean'], 'Hits@10': ours.loc['Hits@10', f'{k} (Centralized) mean']})
display(pd.DataFrame(rows))''')

# ---------------------------------------------------------------------------
add_md(r"""# 12. Communication: measured serialisation vs theoretical estimate
`uplink` is the **measured size of one client's serialised float32 delta** in this process. It is
**not** network traffic: it excludes transport/encryption overhead, mask/key exchange, retries and
dropouts, none of which exist in this simulation.""")
add_code(r'''comm_rows = []
for kind in KINDS:
    params = COMM[kind][0][0]
    measured = float(np.mean([m for _, m in COMM[kind]]))
    comm_rows.append({'model': kind, 'parameters': params, 'theoretical_payload_MB(4B/param)': params * 4 / 2 ** 20,
                      'measured_serialised_uplink_MB/client/round': measured / 2 ** 20,
                      'estimated_total_uplink_MB (clients x rounds)': measured * NUM_CLIENTS * FED_ROUNDS / 2 ** 20})
comm_df = pd.DataFrame(comm_rows); display(comm_df)
print("Downlink (broadcast of the global model) is not counted; these are payload sizes, not measured network traffic.")''')

# ---------------------------------------------------------------------------
add_md(r"""# 13. Differential-privacy illustration (optional, GraphSAGE, first seed)
This shows the *mechanism and accountant*, not a privacy claim for the main results (which use
`DP_SIGMA = 0`). Privacy unit = one client's whole local dataset; with only 4 clients per round
there is no subsampling amplification and the noise needed for a small epsilon swamps the signal.""")
add_code(r'''DP_DEMO = []
if RUN_DP_DEMO:
    base = RESULTS['GraphSAGE (Federated)'][0]
    DP_DEMO.append({'dp_sigma': 0.0, 'epsilon': float('inf'), 'ROC-AUC': base['ROC-AUC'], 'MRR': base['MRR']})
    for sigma in (0.1, 0.5, 1.0, 2.0):
        r = fit_federated('GraphSAGE', ctx0, SEEDS[0], dp_sigma=sigma, clip_norm=DP_CLIP_NORM)
        DP_DEMO.append({'dp_sigma': sigma, 'epsilon': dp_epsilon(sigma, FED_ROUNDS, DP_DELTA),
                        'ROC-AUC': r['test']['ROC-AUC'], 'MRR': r['test']['MRR']})
    dp_df = pd.DataFrame(DP_DEMO); display(dp_df)
    dp_df.to_csv('dp_demo_results.csv', index=False)
    print(f"epsilon is at delta={DP_DELTA}, {FED_ROUNDS} rounds, client-level, all clients every round, no subsampling. "
          "Rows with a large epsilon give no meaningful privacy guarantee. The main results above use DP OFF.")''')

# ---------------------------------------------------------------------------
add_md("# 14. Plots")
add_code(r'''os.makedirs('plots', exist_ok=True)
COL = {'GraphSAGE': '#2c7fb8', 'GAT': '#d95f0e'}

def _despine(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# 1 -- federated validation curves (first seed)
fig, ax1 = plt.subplots(figsize=(9, 4.6), dpi=120); ax2 = ax1.twinx()
for kind in KINDS:
    h = pd.DataFrame(HISTORY[kind]['federated'])
    ax1.plot(h['step'], h['val_auc'], color=COL[kind], lw=2.2, marker='o', ms=3.5, label=f'{kind} val ROC-AUC')
    ax2.plot(h['step'], h['val_mrr'], color=COL[kind], lw=1.6, ls='--', marker='s', ms=3, label=f'{kind} val MRR')
ax1.set_xlabel('Federated round'); ax1.set_ylabel('Validation ROC-AUC'); ax2.set_ylabel('Validation MRR (dashed)')
ax1.grid(alpha=.25, ls='--')
l1, n1 = ax1.get_legend_handles_labels(); l2, n2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, n1 + n2, loc='lower right', fontsize=8)
ax1.set_title(f'Federated training (FedAvg, simulated masking, seed {SEEDS[0]})', fontweight='bold')
fig.tight_layout(); fig.savefig('plots/fed_training_history.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# 2 -- model comparison with seed-level error bars
cols = [f'{k} ({s})' for k in KINDS for s in ('Centralized', 'Federated')]
bar_col = {'GraphSAGE (Centralized)': '#2c7fb8', 'GraphSAGE (Federated)': '#7fcdbb', 'GAT (Centralized)': '#d95f0e', 'GAT (Federated)': '#fdb863'}
sm = summary.set_index('Metric')
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=120)
for ax, metric in zip(axes, ['ROC-AUC', 'PR-AUC', 'MRR']):
    means = [sm.loc[metric, f'{c} mean'] for c in cols]; stds = [sm.loc[metric, f'{c} std'] for c in cols]
    bars = ax.bar(range(len(cols)), means, yerr=stds, capsize=4, color=[bar_col[c] for c in cols], edgecolor='white')
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.replace(' (', '\n(') for c in cols], fontsize=8)
    ax.set_title(f'{metric} (mean ± std, {len(SEEDS)} seeds)', fontweight='bold'); ax.grid(axis='y', alpha=.3, ls='--'); _despine(ax)
    if metric == 'ROC-AUC': ax.axhline(0.5, color='grey', ls='--', lw=.8)
fig.tight_layout(); fig.savefig('plots/model_comparison.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# 3 -- client data distribution
edges_pc = ctx0['stats']['local_train_edges_per_client']
nodes_pc = [int((NODE_TO_CLIENT == c).sum()) for c in range(NUM_CLIENTS)]
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.4), dpi=120)
cc = ['#2c7fb8', '#41b6c4', '#7fcdbb', '#d9ef8b']
a1.bar(range(1, NUM_CLIENTS + 1), nodes_pc, color=cc, edgecolor='white'); a1.set_title('Nodes owned per client', fontweight='bold')
a2.bar(range(1, NUM_CLIENTS + 1), edges_pc, color=cc, edgecolor='white'); a2.set_title('Local (intra-client) training pairs', fontweight='bold')
sh = [ctx0['stats'][k] for k in ('cross_client_train', 'cross_client_val', 'cross_client_test')]
a3.bar(['train', 'val', 'test'], sh, color='#d95f0e', edgecolor='white'); a3.set_ylim(0, 1); a3.set_title('Cross-client share of pairs\n(never used in federated training)', fontweight='bold')
for ax in (a1, a2, a3): ax.grid(axis='y', alpha=.3, ls='--'); _despine(ax)
for ax in (a1, a2): ax.set_xlabel('Client')
fig.tight_layout(); fig.savefig('plots/client_data_distribution.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# 4 -- communication (measured serialised payload; no network)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), dpi=120)
x = np.arange(len(comm_df)); w = .35
a1.bar(x - w/2, comm_df['theoretical_payload_MB(4B/param)'], w, label='theoretical (4 B/param)', color='#2c7fb8')
a1.bar(x + w/2, comm_df['measured_serialised_uplink_MB/client/round'], w, label='measured pickle size', color='#d95f0e')
a1.set_xticks(x); a1.set_xticklabels(comm_df['model']); a1.set_ylabel('MB per client per round'); a1.legend(fontsize=8)
a1.set_title('Client uplink payload', fontweight='bold')
for kind in KINDS:
    per_round = comm_df.set_index('model').loc[kind, 'measured_serialised_uplink_MB/client/round'] * NUM_CLIENTS
    a2.plot(range(1, FED_ROUNDS + 1), np.cumsum(np.full(FED_ROUNDS, per_round)), color=COL[kind], lw=2.2, label=kind)
a2.set_xlabel('Federated round'); a2.set_ylabel('Cumulative uplink (MB, all clients)'); a2.legend(); a2.set_title('Cumulative uplink payload (estimate)', fontweight='bold')
for ax in (a1, a2): ax.grid(alpha=.3, ls='--'); _despine(ax)
fig.tight_layout(); fig.savefig('plots/communication_analysis.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# 5 -- dataset structure (undirected-pair degree distribution + locations)
deg = np.bincount(ALL_PAIRS.reshape(-1), minlength=num_nodes)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=120)
vals, cnts = np.unique(deg, return_counts=True)
a1.loglog(vals[vals > 0], cnts[vals > 0], 'o', ms=3.5, color='#2c7fb8'); a1.set_xlabel('Degree (undirected pairs)'); a1.set_ylabel('Nodes')
a1.set_title('Degree distribution (log-log)', fontweight='bold'); a1.grid(alpha=.3, ls='--'); _despine(a1)
loc = pd.Series(node_locations).value_counts().head(12)
a2.barh(range(len(loc)), loc.values, color='#d95f0e', edgecolor='white'); a2.set_yticks(range(len(loc))); a2.set_yticklabels(loc.index, fontsize=8)
a2.invert_yaxis(); a2.set_title('Top 12 locations (incl. "Unknown" = missing)', fontweight='bold'); a2.grid(axis='x', alpha=.3, ls='--'); _despine(a2)
fig.tight_layout(); fig.savefig('plots/dataset_structure.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print("plots written to plots/")''')

# ---------------------------------------------------------------------------
add_md(r"""# 15. Node classification: synthetic labels only (**no classifier is trained here**)
The reference paper also describes node classification on AnonNet using *synthetic* carbon scores
(low/medium/high risk). This cell only generates such synthetic labels. They are a deterministic
function of `location` plus noise, so they are **not** real sustainability data, and **no node-classification result is
reported by this project.**""")
add_code(r'''def generate_carbon_scores(df, n, seed):
    rng = np.random.default_rng(seed)
    intensity = {'China': 1.00, 'India': 0.85, 'Japan': 0.55, 'South Korea': 0.60, 'United States': 0.70,
                 'Germany': 0.45, 'United Kingdom': 0.40, 'Taiwan': 0.65, 'Indonesia': 0.90, 'Australia': 0.60, 'Thailand': 0.75}
    locs = df['location'].fillna('Unknown').astype(str).tolist()
    return np.array([intensity.get(l, 0.65) * rng.lognormal(sigma=0.35) for l in locs], dtype=np.float32)

carbon = generate_carbon_scores(nodes_df, num_nodes, SEEDS[0])
p33, p67 = np.percentile(carbon, [33.33, 66.67])
risk = np.where(carbon <= p33, 0, np.where(carbon <= p67, 1, 2))
print("synthetic risk-band counts (Low/Medium/High):", np.bincount(risk, minlength=3).tolist(), "-- illustrative only")''')

# ---------------------------------------------------------------------------
add_md("# 16. Run metadata (reproducibility record)")
add_code(r'''meta = {
    'notebook': 'anonnet_federated_kg_completion.ipynb',
    'run_timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    'python': platform.python_version(), 'torch': torch.__version__,
    'torch_geometric': torch_geometric.__version__ if _HAS_PYG else None,
    'numpy': np.__version__, 'pandas': pd.__version__,
    'seeds': SEEDS,
    'config': {k: globals()[k] for k in ['HIDDEN_DIM', 'DROPOUT', 'LEARNING_RATE', 'SUP_FRAC', 'NUM_CLIENTS', 'LOCAL_EPOCHS',
                                        'FED_ROUNDS', 'CENTRAL_EPOCHS', 'NEG_RATIO', 'RANKING_CANDIDATES', 'SPLIT',
                                        'PAIRWISE_MASK_SIM', 'DP_SIGMA', 'DP_CLIP_NORM', 'DP_DELTA']},
    'privacy_status': {'differential_privacy': 'OFF (DP_SIGMA=0): no (epsilon, delta) guarantee in the reported runs',
                       'secure_aggregation': 'simulated pairwise masking in one process; not a deployable protocol'},
    'evaluation_protocol': pool_note,
    'data': {'nodes': num_nodes, 'undirected_pairs': int(len(ALL_PAIRS)), 'relation_column': RELATION_COL},
    'per_seed_split_stats': {str(k): v for k, v in STATS_PER_SEED.items()},
    'leakage_acceptance_tests': 'passed in prepare() for every seed (disjoint splits; no held-out pair/reverse in any message graph; client isolation)',
}
with open('run_metadata_anonnet.json', 'w') as f:
    json.dump(meta, f, indent=2, default=str)
print(json.dumps({k: meta[k] for k in ['run_timestamp_utc', 'torch', 'torch_geometric', 'seeds']}, indent=1))''')

# ---------------------------------------------------------------------------
add_md(r"""# 17. Interpretation and Claims

## What the results can support
- Comparison of *centralized* vs *simulated-federated* training of GraphSAGE/GAT for **homogeneous link
  prediction** on AnonNet, under a leakage-free protocol, as mean ± std over several seeds.
- A non-learned degree-product baseline is reported next to the GNNs; read the GNN numbers relative to it, because uniform random negatives
  favour low-degree nodes and degree alone separates a lot.
- The federated models learn only from intra-client pairs; the measured gap to centralized training
  therefore reflects *both* the loss of cross-client structure *and* the feature/degree distribution shift between local
  and global graphs. This notebook does not separate the two effects.

## What the results do **not** support
- **No privacy guarantee.** DP is off. The masking is a one-process arithmetic simulation, so it is not evidence that
  a real server cannot inspect individual updates; it is not a Secure Aggregation implementation.
- **No deployment claim.** Clients are simulated partitions of one dataset in one process.
- **No claim that the GNNs learn more than degree structure.** In the reported run the non-learned degree-product baseline matched or exceeded the
  GNNs on ROC-AUC and PR-AUC (see the results table). With uniform random negatives, a degree prior is very strong on this graph; the GNNs
  are not shown to add predictive value beyond it.
- **No relation-aware KGC claim.** AnonNet has no relation labels.
- **No claim of beating the reference paper**: protocols differ (candidate pools, negatives, splits).
- Ranking metrics depend on the sampled candidate pool (here 50 filtered candidates) and are not comparable
  across papers using other pools. Thresholded metrics use a threshold tuned on validation data;
  with 1 : 5 class balance the majority-class accuracy is already 0.833, so look at balanced accuracy, ROC-AUC and PR-AUC.

## Known limitations
- Cross-client pairs (see section 9 for the share) are excluded from federated training.
- Region-based ownership gives clients with very different sizes and a large `Unknown`-location block.
- Degree features computed on a client's local graph have a different scale from degrees on the global training graph.
- Communication numbers are payload sizes, not measured network traffic.

## Future work (not implemented)
Real Secure Aggregation inside a federated framework; calibrated DP with larger client counts; relation labels for
multi-relational GNNs; learned cross-silo link models; measured multi-process communication.""")

n_cells = nb.write("anonnet_federated_kg_completion.ipynb")
print(f"Notebook written: anonnet_federated_kg_completion.ipynb ({n_cells} cells)")
