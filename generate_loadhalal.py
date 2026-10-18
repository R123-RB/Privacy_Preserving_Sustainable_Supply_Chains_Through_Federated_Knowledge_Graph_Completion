"""Generates loadhalal_federated_kg_completion.ipynb  (corrected version, see AUDIT_FIXES.md).

Run:  python generate_loadhalal.py
"""
import notebook_common as nc

nb = nc.Notebook()
add_md, add_code = nb.md, nb.code

add_md(r"""# 1. Project Overview
**Simulated federated multi-relational link prediction on the LoadHalal food supply graph** (research prototype).

Task: predict the `has_manufacturer` link (food -> manufacturer) using a relational GCN (RGCN) over three relations
(`has_manufacturer`, `contains`, `halal_status`). Clients are simulated manufacturers/data holders.

**What this notebook is, precisely**
- A *single-process simulation* of cross-silo FL: no network, authentication or real isolation.
- Aggregation is a **simulation of additive pairwise masking** (arithmetic only) — *not* a deployable Secure Aggregation protocol.
- **Differential privacy is OFF** (`DP_SIGMA = 0`): no (epsilon, delta) guarantee is claimed.
- **Leakage-free protocol.** The `has_manufacturer` edges are split into train/val/test *first*. Held-out target edges
  are removed from every message-passing graph and from every degree feature. The previous version passed the full
  graph (including all held-out target edges) to the encoder *and* mis-oriented its edge tensors, so its reported
  perfect scores were not valid evidence of model quality (see `AUDIT_FIXES.md`).""")

add_code(nc.IMPORTS_CELL)

add_md("# 2. Configuration")
add_code(r'''def _find_dataset_dir():
    from pathlib import Path
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    candidates.append(Path.cwd()); candidates.extend(Path.cwd().parents)
    for base in candidates:
        if (base / "Loadhalal dataset" / "food.csv").exists():
            return base
    return Path(".")

DATA_DIR = str(_find_dataset_dir() / "Loadhalal dataset")
PRIMARY_RELATION = "has_manufacturer"          # the link-prediction target

# --- Model / training ---
HIDDEN_DIM = 64
DROPOUT = 0.2
LEARNING_RATE = 0.01
NUM_BASES = 4
SUP_FRAC = 0.3             # per epoch, share of TRAIN target edges hidden from message passing and used as supervision
ID_FEATURE_DIM = 32        # fixed random identity vector per node (public, seed-independent; NOT learned, no per-node parameters).
                           # Without identity features a GNN cannot tell two certificates apart; 0 disables them (ablation, section 11).
RUN_ID_ABLATION = True

# --- Federation ---
NUM_CLIENTS = 4
LOCAL_EPOCHS = 3
FED_ROUNDS = 20
CENTRAL_EPOCHS = FED_ROUNDS * LOCAL_EPOCHS

# --- Evaluation protocol ---
NEG_RATIO = 5
RANKING_CANDIDATES = 50    # filtered tail corruptions (manufacturers) per positive
SPLIT = (0.8, 0.1, 0.1)
SEEDS = [42, 43, 44]

# --- Privacy-related switches ---
PAIRWISE_MASK_SIM = True   # SIMULATED pairwise masking (arithmetic only)
DP_SIGMA = 0.0             # 0.0 => differential privacy OFF
DP_CLIP_NORM = 1.0
DP_DELTA = 1e-5

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEEDS[0])
print(f"seeds={SEEDS} | hidden={HIDDEN_DIM} | id features={ID_FEATURE_DIM} | DP {'ON' if DP_SIGMA > 0 else 'OFF'} | masking simulation={PAIRWISE_MASK_SIM}")''')

add_md(r"""# 3. Load and Validate the Data
`manufacturer.csv` contains **unquoted commas inside names** (e.g. `356,SRIBOGA BAKERIES INTEGRA, PT`). Selecting the first two
columns with `usecols` silently truncates such names, so the file is parsed explicitly (split on the *first* comma only)
and validated: id format, uniqueness, row count, and agreement with the naive parser on every id.""")
add_code(r'''def read_manufacturer(path):
    rows, bad, n_extra_commas = [], [], 0
    with open(path, encoding='utf-8') as f:
        header = f.readline().rstrip('\r\n').split(',')
        assert header[0] == 'id', f"unexpected header {header}"
        for ln, line in enumerate(f, start=2):
            line = line.rstrip('\r\n')
            if not line:
                continue
            idpart, _, name = line.partition(',')
            if not idpart.strip().isdigit():
                bad.append((ln, line)); continue
            n_extra_commas += (',' in name)
            rows.append((int(idpart), name.strip().strip('"')))
    return pd.DataFrame(rows, columns=['id', 'nama_man']), bad, n_extra_commas

food_df = pd.read_csv(f"{DATA_DIR}/food.csv")
ingredient_df = pd.read_csv(f"{DATA_DIR}/ingredient.csv")
certificate_df = pd.read_csv(f"{DATA_DIR}/certificate.csv")
manufacturer_df, bad_rows, n_comma_names = read_manufacturer(f"{DATA_DIR}/manufacturer.csv")
naive = pd.read_csv(f"{DATA_DIR}/manufacturer.csv", usecols=[0, 1])

# ---- parser validation ----
n_lines = sum(1 for _ in open(f"{DATA_DIR}/manufacturer.csv", encoding='utf-8')) - 1
assert not bad_rows, f"malformed manufacturer rows: {bad_rows[:3]}"
assert len(manufacturer_df) == n_lines, "row count mismatch"
assert manufacturer_df['id'].is_unique, "duplicate manufacturer ids"
assert set(manufacturer_df['id']) == set(naive['id']), "robust and naive parsers disagree on ids"
truncated = int((manufacturer_df.set_index('id')['nama_man'] != naive.set_index('id')['nama_man'].reindex(manufacturer_df['id']).values).sum())
print(f"manufacturer.csv: {len(manufacturer_df)} rows, unique ids OK, {n_comma_names} names contain commas; "
      f"{truncated} names would have been truncated by the usecols workaround")
for name, df in [('food', food_df), ('ingredient', ingredient_df), ('certificate', certificate_df)]:
    assert df['id'].is_unique, f"duplicate ids in {name}"
    print(f"{name:12s}: {len(df):6d} rows, unique ids OK")

fm_df = pd.read_csv(f"{DATA_DIR}/food_manufacturer.csv"); fi_df = pd.read_csv(f"{DATA_DIR}/food_ingredient.csv"); fc_df = pd.read_csv(f"{DATA_DIR}/food_certificate.csv")
print("edge files:", {'food_manufacturer': fm_df.shape, 'food_ingredient': fi_df.shape, 'food_certificate': fc_df.shape})''')

add_md(r"""# 4. Graph Construction
Nodes: foods, manufacturers, ingredients, certificates. Edges are keyed by the raw integer ids of each *type*
(ids are only unique within a type). Edges whose endpoints are missing from the node tables are dropped and **reported**, and
duplicates are removed.""")
add_code(r'''node_tables = [('food', food_df), ('manufacturer', manufacturer_df), ('ingredient', ingredient_df), ('certificate', certificate_df)]
RANGES, TYPE_INDEX, offset = {}, {}, 0
for t, df in node_tables:
    RANGES[t] = (offset, offset + len(df))
    TYPE_INDEX[t] = {int(i): offset + k for k, i in enumerate(df['id'])}
    offset += len(df)
num_nodes = offset
FOOD_RANGE, MFR_RANGE = RANGES['food'], RANGES['manufacturer']
FOODS, MFRS = np.arange(*FOOD_RANGE), np.arange(*MFR_RANGE)
node_type = np.concatenate([np.full(hi - lo, k) for k, (t, (lo, hi)) in enumerate(RANGES.items())])
print("node ranges:", RANGES, "| total nodes:", num_nodes)

REL_NAMES = ['has_manufacturer', 'contains', 'halal_status']
edge_specs = [(fm_df, 'food', 'manufacturer'), (fi_df, 'food', 'ingredient'), (fc_df, 'food', 'certificate')]
REL_EDGES, EDGE_QUALITY = [], []
for name, (df, st, dt) in zip(REL_NAMES, edge_specs):
    s_raw, d_raw = df.iloc[:, 0].to_numpy(np.int64), df.iloc[:, 1].to_numpy(np.int64)
    s = np.array([TYPE_INDEX[st].get(int(v), -1) for v in s_raw]); d = np.array([TYPE_INDEX[dt].get(int(v), -1) for v in d_raw])
    ok = (s >= 0) & (d >= 0)
    pairs = np.unique(np.stack([s[ok], d[ok]], 1), axis=0)
    EDGE_QUALITY.append({'relation': name, 'raw_rows': len(df), 'dropped_missing_endpoint': int((~ok).sum()),
                         'dropped_duplicates': int(ok.sum() - len(pairs)), 'kept': len(pairs)})
    REL_EDGES.append(pairs)
quality_df = pd.DataFrame(EDGE_QUALITY); display(quality_df)
for q in EDGE_QUALITY:
    if q['dropped_missing_endpoint'] / max(q['raw_rows'], 1) > 0.05:
        print(f"DATA-QUALITY WARNING: {q['dropped_missing_endpoint'] / q['raw_rows']:.1%} of '{q['relation']}' rows reference ids absent from the node table "
              f"and are discarded (not a parsing artefact: the ids are genuinely missing from the source tables).")

TARGET = REL_EDGES[0]                       # [E, 2] (food, manufacturer)
OTHER = REL_EDGES[1:]                       # contains, halal_status
fpm = pd.Series(TARGET[:, 0]).value_counts()
print(f"target edges: {len(TARGET)} | foods with a target edge: {len(fpm)} | max manufacturers per food: {int(fpm.max())}")
assert (TARGET[:, 0] >= FOOD_RANGE[0]).all() and (TARGET[:, 0] < FOOD_RANGE[1]).all()
assert (TARGET[:, 1] >= MFR_RANGE[0]).all() and (TARGET[:, 1] < MFR_RANGE[1]).all()
halal_foods = np.unique(OTHER[1][:, 0])
print(f"foods with >= 1 halal certificate edge: {len(halal_foods)} / {len(food_df)}")''')

add_code(nc.METRICS_CELL)
add_code(nc.SAMPLING_CELL)
add_md(r"""# 5. Aggregation: simulated pairwise masking + optional client-level DP
> **Scope warning.** The code below *simulates* additive pairwise masking inside one process and tests that masks cancel.
> It is **not** a Secure Aggregation protocol (no key agreement, authentication, dropout recovery or adversary model) and does
> **not** show that a real server could not inspect an individual update. DP is **off** unless `DP_SIGMA > 0`.
> Clients send model **deltas**, not raw gradients.""")
add_code(nc.AGGREGATION_CELL)

add_md(r"""# 6. Model — Relational GCN
Messages flow **src -> dst**. For every permitted edge `(u, v, r)` an *inverse* relation `(v, u, r+3)` is added (standard RGCN
practice), so manufacturers also receive information from their foods. Inverse edges are only ever created for edges that are
allowed in the message-passing graph, so held-out target edges never appear in either direction.
Per-relation weights use basis decomposition; aggregation is a per-relation mean. Hand-rolled (no `torch-sparse` dependency).""")
add_code(r'''NUM_REL = 3
R_TOTAL = 2 * NUM_REL                       # forward + inverse relation types

class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rel, num_bases):
        super().__init__()
        self.num_rel = num_rel
        self.basis = nn.Parameter(torch.empty(num_bases, in_dim, out_dim))
        self.coef = nn.Parameter(torch.empty(num_rel, num_bases))
        self.w_self = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        for p in (self.basis, self.coef, self.w_self):
            nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, edge_type):
        N = x.size(0)
        W = torch.einsum('rb,bio->rio', self.coef, self.basis)
        out = x @ self.w_self + self.bias
        src, dst = edge_index
        for r in range(self.num_rel):
            m = edge_type == r
            if not bool(m.any()):
                continue
            s, d = src[m], dst[m]
            agg = torch.zeros(N, out.size(1)).index_add_(0, d, x[s] @ W[r])
            cnt = torch.bincount(d, minlength=N).clamp(min=1).unsqueeze(1).float()
            out = out + agg / cnt
        return out

class RGCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_rel, dropout, num_bases):
        super().__init__()
        self.l1 = RGCNLayer(in_dim, hidden_dim, num_rel, num_bases)
        self.l2 = RGCNLayer(hidden_dim, hidden_dim, num_rel, num_bases)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        h = self.dropout(F.relu(self.l1(x, edge_index, edge_type)))
        return self.l2(h, edge_index, edge_type)          # linear output (no final ReLU)

class LinkPredictor(nn.Module):
    """Cosine similarity x learnable temperature. Input: pairs LongTensor [E, 2] = (food, manufacturer)."""
    def __init__(self, init_tau=1.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(init_tau)))
    def forward(self, z, pairs):
        assert pairs.dim() == 2 and pairs.size(1) == 2, f"pairs must be [E, 2], got {tuple(pairs.shape)}"
        z = F.normalize(z, p=2, dim=-1)
        return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(-1) * self.tau

class KGCModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = RGCNEncoder(in_dim, HIDDEN_DIM, R_TOTAL, DROPOUT, NUM_BASES)
        self.predictor = LinkPredictor()

def make_model(in_dim):
    return KGCModel(in_dim)
print("RGCN parameters:", sum(p.numel() for p in make_model(10 + ID_FEATURE_DIM).parameters()))''')

add_md(r"""# 7. Graph helpers, features, training step, evaluation
`build_msg_graph` assembles a message-passing graph from *explicitly passed* edge sets. `make_x` computes per-relation log-degree
features **from that same graph**, so a feature can never encode an edge that is not in the graph the model sees.""")
add_code(r'''def build_msg_graph(target_pairs, other_pairs):
    """target_pairs [E,2] (the ONLY target-relation edges allowed in this graph) + other relations
    -> (edge_index [2, E_total], edge_type [E_total]) with inverse relations added."""
    src, dst, et = [], [], []
    for r, pairs in enumerate([target_pairs] + list(other_pairs)):
        if pairs.numel() == 0:
            continue
        src += [pairs[:, 0], pairs[:, 1]]; dst += [pairs[:, 1], pairs[:, 0]]
        et += [torch.full((pairs.size(0),), r), torch.full((pairs.size(0),), r + NUM_REL)]
    if not src:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    return torch.stack([torch.cat(src), torch.cat(dst)]), torch.cat(et)

def make_x(static_x, edge_index, edge_type, num_nodes):
    if edge_index.numel():
        deg = torch.bincount(edge_index[1] * R_TOTAL + edge_type, minlength=num_nodes * R_TOTAL).view(num_nodes, R_TOTAL).float()
    else:
        deg = torch.zeros(num_nodes, R_TOTAL)
    return torch.cat([static_x, torch.log1p(deg)], dim=1)

def train_step(model, opt, static_x, num_nodes, target_pairs, other_pairs, neg_pool, gen):
    E = target_pairs.size(0)
    if E < 2:
        return float('nan')
    model.train(); opt.zero_grad()
    perm = torch.randperm(E, generator=gen)
    n_sup = max(1, int(round(SUP_FRAC * E)))
    sup, msg = target_pairs[perm[:n_sup]], target_pairs[perm[n_sup:]]
    ei, et = build_msg_graph(msg, other_pairs)
    z = model.encoder(make_x(static_x, ei, et, num_nodes), ei, et)
    neg = neg_pool[torch.randint(0, neg_pool.size(0), (NEG_RATIO * n_sup,), generator=gen)]
    ps, ns = model.predictor(z, sup), model.predictor(z, neg)
    loss = F.binary_cross_entropy_with_logits(ps, torch.ones_like(ps)) + F.binary_cross_entropy_with_logits(ns, torch.zeros_like(ns))
    loss.backward(); opt.step()
    return float(loss.detach())

@torch.no_grad()
def evaluate_model(model, ctx, split, thr=None):
    """Scores val/test on the graph made of TRAIN target edges + the non-target relations (never held-out target edges)."""
    model.eval()
    ei, et = ctx['eval_graph']
    z = F.normalize(model.encoder(make_x(ctx['static_x'], ei, et, ctx['num_nodes']), ei, et), dim=-1)
    tau = model.predictor.tau
    sc = lambda p: ((z[p[:, 0]] * z[p[:, 1]]).sum(-1) * tau).cpu().numpy()
    b = ctx[split]
    pos, neg = sc(b['pos']), sc(b['neg'])
    P, K = b['cand'].shape[:2]
    cand = sc(b['cand'].reshape(-1, 2)).reshape(P, K)
    thr = best_f1_threshold(pos, neg) if thr is None else thr
    return summarize_scores(pos, neg, pos, cand, thr)

def popularity_baseline(ctx):
    """NON-LEARNED reference: score(food, mfr) = log(1 + number of TRAIN foods of that manufacturer).
    Positives are drawn in proportion to manufacturer size while sampled negatives are uniform, so this
    ignores the food entirely yet can score well; the GNN must beat it to show it learned anything more."""
    deg = np.bincount(ctx['train_pairs'][:, 1].numpy(), minlength=num_nodes)
    sc = lambda p: np.log1p(deg[p[:, 1].numpy()]).astype(np.float64)
    def run(split, thr=None):
        b = ctx[split]; pos, neg = sc(b['pos']), sc(b['neg']); P, K = b['cand'].shape[:2]
        cand = sc(b['cand'].reshape(-1, 2)).reshape(P, K)
        return summarize_scores(pos, neg, pos, cand, best_f1_threshold(pos, neg) if thr is None else thr)
    return run('test', run('val')['Threshold'])''')

add_md(r"""# 8. Federated Ownership and Per-Experiment Preparation
**Ownership rule (label-independent).** Manufacturers are assigned to clients round-robin by id. A food belongs to the client of its
manufacturer *only if that edge is a TRAIN edge*; foods whose target edge is held out (validation/test) get an arbitrary owner
(`id mod K`) that is independent of their true manufacturer. (The previous version derived ownership from *all* target edges, including
held-out ones.) Ingredients/certificates follow the majority owner of the foods they link to.
Because each food has exactly one manufacturer, **100% of train target edges are intra-client by construction** (a manufacturer owns
its own supply edges); cross-client structure appears in the *non-target* relations and in validation/test edges, and is reported below.

A client sees only: its owned nodes' static rows, edges with both endpoints owned, local typed negatives (its foods x its manufacturers)
and degree features computed from its local graph.""")
add_code(r'''def assert_no_heldout_target(ei, et, heldout, name):
    """Held-out (food, manufacturer) pairs must be absent from the forward AND inverse target relation."""
    m_fwd, m_inv = et == 0, et == NUM_REL
    fwd = make_key_array(torch.stack([ei[0][m_fwd], ei[1][m_fwd]], 1).numpy(), num_nodes, False)
    inv = make_key_array(torch.stack([ei[1][m_inv], ei[0][m_inv]], 1).numpy(), num_nodes, False)     # inverse stored as (mfr, food)
    h = make_key_array(heldout, num_nodes, False)
    assert not np.isin(h, fwd).any() and not np.isin(h, inv).any(), f"LEAK: held-out target edge in {name}"

def assign_owners(train_pairs, heldout_foods):
    K = NUM_CLIENTS
    owner = -np.ones(num_nodes, dtype=np.int64)
    owner[MFRS] = (MFRS - MFR_RANGE[0]) % K
    owner[train_pairs[:, 0]] = owner[train_pairs[:, 1]]                       # food <- its TRAIN manufacturer
    unowned = FOODS[owner[FOODS] < 0]
    owner[unowned] = unowned % K                                              # label-independent
    for pairs in OTHER:                                                       # ingredients / certificates: majority of linked foods
        votes = np.zeros((num_nodes, K), dtype=np.int64)
        np.add.at(votes, (pairs[:, 1], owner[pairs[:, 0]]), 1)
        has = votes.sum(1) > 0
        idx = np.where(has & (owner < 0))[0]
        owner[idx] = votes[idx].argmax(1)
    rest = np.where(owner < 0)[0]
    owner[rest] = rest % K
    assert (owner >= 0).all()
    return owner

def assert_client_isolation(c, owner):
    n = c['num_nodes']
    assert c['static_x'].shape[0] == n, "client holds features for nodes it does not own"
    G = c['nodes_global']
    assert (owner[G] == c['client_id']).all()
    for name in ['target_pairs', 'neg_pool'] + [f'other{i}' for i in range(len(OTHER))]:
        t = c[name] if name in c else c['other'][int(name[-1])]
        assert t.numel() == 0 or (t.ndim == 2 and t.shape[1] == 2 and int(t.min()) >= 0 and int(t.max()) < n), f"{name} out of range"
    for t in (c['target_pairs'], c['neg_pool']):
        assert t.numel() == 0 or ((node_type[G[t[:, 0].numpy()]] == 0).all() and (node_type[G[t[:, 1].numpy()]] == 1).all()), "pairs must be food -> manufacturer"

def build_clients(ctx, owner, seed):
    tr = ctx['train_pairs'].numpy()
    clients = []
    for c in range(NUM_CLIENTS):
        nodes_c = np.where(owner == c)[0]
        g2l = -np.ones(num_nodes, dtype=np.int64); g2l[nodes_c] = np.arange(len(nodes_c))
        local_t = tr[(owner[tr[:, 0]] == c) & (owner[tr[:, 1]] == c)]
        local_other = [p[(owner[p[:, 0]] == c) & (owner[p[:, 1]] == c)] for p in OTHER]
        foods_c = g2l[np.unique(local_t[:, 0])] if len(local_t) else np.zeros(0, dtype=np.int64)
        mfrs_c = g2l[MFRS[owner[MFRS] == c]]
        rng = np.random.default_rng(seed * 100 + 10 + c)
        lt = g2l[local_t]
        neg = sample_negative_pairs(max(NEG_RATIO * len(lt) * 2, 16), rng, foods_c, mfrs_c, make_key_array(lt, len(nodes_c), False), len(nodes_c), False)
        cl = {'client_id': c, 'nodes_global': nodes_c, 'num_nodes': len(nodes_c), 'static_x': ctx['static_x'][nodes_c].clone(),
              'target_pairs': torch.LongTensor(lt), 'other': [torch.LongTensor(g2l[p]) for p in local_other], 'neg_pool': torch.LongTensor(neg)}
        assert_client_isolation(cl, owner)
        clients.append(cl)
    return clients

def make_static(id_dim):
    onehot = torch.zeros(num_nodes, 4); onehot[torch.arange(num_nodes), torch.as_tensor(node_type)] = 1.0
    if id_dim <= 0:
        return onehot
    g = torch.Generator().manual_seed(12345)           # fixed, public, independent of seed/split/labels
    return torch.cat([onehot, torch.randn(num_nodes, id_dim, generator=g)], 1)

def prepare(seed, id_dim=None):
    id_dim = ID_FEATURE_DIM if id_dim is None else id_dim
    rng = np.random.default_rng(seed)
    P = len(TARGET); perm = rng.permutation(P)
    n_tr, n_va = int(SPLIT[0] * P), int(SPLIT[1] * P)
    train, val, test = TARGET[perm[:n_tr]], TARGET[perm[n_tr:n_tr + n_va]], TARGET[perm[n_tr + n_va:]]
    k_all, k_tr = make_key_array(TARGET, num_nodes, False), make_key_array(train, num_nodes, False)
    for a, b in ((train, val), (train, test), (val, test)):
        assert not np.isin(make_key_array(a, num_nodes, False), make_key_array(b, num_nodes, False)).any(), "splits overlap"
    # training negatives: only foods that are TRAIN-positive heads (held-out foods must not be taught as 'always negative')
    train_neg = sample_negative_pairs(NEG_RATIO * len(train) * 2, rng, np.unique(train[:, 0]), MFRS, k_tr, num_nodes, False)
    def bundle(pairs):
        neg = sample_negative_pairs(NEG_RATIO * len(pairs), rng, FOODS, MFRS, k_all, num_nodes, False)
        cand = build_ranking_candidates(pairs, RANKING_CANDIDATES, rng, MFRS, k_all, num_nodes, False, 'tail')
        assert not in_keys(k_all, make_key_array(neg, num_nodes, False)).any()
        return {'pos': torch.LongTensor(pairs), 'neg': torch.LongTensor(neg), 'cand': torch.LongTensor(cand)}
    static_x = make_static(id_dim)
    other_t = [torch.LongTensor(p) for p in OTHER]
    ctx = {'seed': seed, 'num_nodes': num_nodes, 'static_x': static_x, 'in_dim': static_x.shape[1] + R_TOTAL,
           'train_pairs': torch.LongTensor(train), 'other': other_t, 'train_neg_pool': torch.LongTensor(train_neg),
           'val': bundle(val), 'test': bundle(test)}
    ctx['eval_graph'] = build_msg_graph(ctx['train_pairs'], other_t)

    # ---- acceptance tests ----
    heldout = np.concatenate([val, test])
    assert_no_heldout_target(*ctx['eval_graph'], heldout, "the evaluation message-passing graph")
    sp = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    assert_no_heldout_target(*build_msg_graph(ctx['train_pairs'][sp[int(SUP_FRAC * len(train)):]], other_t), heldout, "a training message-passing graph")
    leaky = build_msg_graph(torch.cat([ctx['train_pairs'], torch.LongTensor(test[:1])]), other_t)
    fired = False
    try:
        assert_no_heldout_target(*leaky, heldout, "deliberately leaky graph")
    except AssertionError:
        fired = True
    assert fired, "leak test is insensitive"
    x_eval = make_x(static_x, *ctx['eval_graph'], num_nodes)
    held_foods = np.unique(heldout[:, 0])
    assert float(x_eval[held_foods][:, -R_TOTAL].abs().max()) == 0.0 and float(x_eval[held_foods][:, -R_TOTAL + NUM_REL].abs().max()) == 0.0, \
        "target-relation degree of a held-out food is non-zero"

    owner = assign_owners(train, held_foods)
    assert (owner[held_foods] == held_foods % NUM_CLIENTS).all(), "held-out foods' ownership must not depend on their label"
    ctx['owner'] = owner
    ctx['clients'] = build_clients(ctx, owner, seed)
    cross = lambda a: float((owner[a[:, 0]] != owner[a[:, 1]]).mean())
    ctx['stats'] = {'train_edges': len(train), 'val_edges': len(val), 'test_edges': len(test),
                    'cross_client_train_target': cross(train), 'cross_client_val_target': cross(val), 'cross_client_test_target': cross(test),
                    'cross_client_contains': cross(OTHER[0]), 'cross_client_halal_status': cross(OTHER[1]),
                    'local_target_edges_per_client': [int(c['target_pairs'].size(0)) for c in ctx['clients']],
                    'nodes_per_client': [int(c['num_nodes']) for c in ctx['clients']]}
    return ctx

ctx0 = prepare(SEEDS[0])
st = ctx0['stats']
print(f"split (seed {SEEDS[0]}): train={st['train_edges']} val={st['val_edges']} test={st['test_edges']} target edges | in_dim={ctx0['in_dim']}")
print(f"cross-client share: train target={st['cross_client_train_target']:.1%} (0% by construction) | val target={st['cross_client_val_target']:.1%} | "
      f"test target={st['cross_client_test_target']:.1%} | contains={st['cross_client_contains']:.1%} | halal_status={st['cross_client_halal_status']:.1%}")
print("nodes per client:", st['nodes_per_client'], "| local train target edges:", st['local_target_edges_per_client'])
print("acceptance tests passed: disjoint splits; no held-out target edge (either direction) in any message graph; held-out foods have zero "
      "target-relation degree; ownership label-independent; client isolation OK")''')

add_md(r"""# 9. Centralized and Federated Training
Same number of gradient steps per client in both settings; best checkpoint chosen on **validation ROC-AUC**; thresholded metrics use a
threshold tuned on validation only. The server scores val/test edges on the global training graph — an *evaluation harness*, not
something any single party would hold in deployment.""")
add_code(r'''def fit_central(ctx, seed):
    set_seed(seed)
    model = make_model(ctx['in_dim']); opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    gen = torch.Generator().manual_seed(seed)
    best, hist = {'auc': -1.0}, []
    for ep in range(1, CENTRAL_EPOCHS + 1):
        loss = train_step(model, opt, ctx['static_x'], ctx['num_nodes'], ctx['train_pairs'], ctx['other'], ctx['train_neg_pool'], gen)
        if ep % LOCAL_EPOCHS == 0:
            m = evaluate_model(model, ctx, 'val')
            hist.append({'step': ep // LOCAL_EPOCHS, 'loss': loss, 'val_auc': m['ROC-AUC'], 'val_mrr': m['MRR']})
            if m['ROC-AUC'] > best['auc']:
                best = {'auc': m['ROC-AUC'], 'state': copy.deepcopy(model.state_dict()), 'step': ep // LOCAL_EPOCHS}
    model.load_state_dict(best['state'])
    thr = evaluate_model(model, ctx, 'val')['Threshold']
    return {'test': evaluate_model(model, ctx, 'test', thr), 'best_step': best['step'], 'history': hist, 'model': model}

def client_update(ctx, global_state, client, seed, rnd):
    model = make_model(ctx['in_dim']); model.load_state_dict(global_state)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    gen = torch.Generator().manual_seed(seed * 100003 + rnd * 101 + client['client_id'])
    n = client['target_pairs'].size(0)
    if n >= 2:
        for _ in range(LOCAL_EPOCHS):
            train_step(model, opt, client['static_x'], client['num_nodes'], client['target_pairs'], client['other'], client['neg_pool'], gen)
    else:
        n = 0
    new = model.state_dict()
    return {k: (new[k] - global_state[k]).float() for k in global_state}, n       # model DELTA

def fit_federated(ctx, seed, dp_sigma=None, clip_norm=None):
    set_seed(seed)
    gm = make_model(ctx['in_dim']); masker = PairwiseMaskSimulation(NUM_CLIENTS, seed)
    best, hist, uplink = {'auc': -1.0}, [], []
    for rnd in range(1, FED_ROUNDS + 1):
        gs = {k: v.clone() for k, v in gm.state_dict().items()}
        results = [client_update(ctx, gs, c, seed, rnd) for c in ctx['clients']]
        agg, info = aggregate_round(results, rnd, masker, dp_sigma=dp_sigma, clip_norm=clip_norm)
        if agg is not None:
            gm.load_state_dict({k: gs[k] + agg[k] for k in gs}); uplink.append(info['uplink_bytes_per_client'])
        m = evaluate_model(gm, ctx, 'val')
        hist.append({'step': rnd, 'val_auc': m['ROC-AUC'], 'val_mrr': m['MRR'], 'val_f1': m['F1-score'], 'val_prauc': m['PR-AUC'], 'val_hits10': m['Hits@10']})
        if m['ROC-AUC'] > best['auc']:
            best = {'auc': m['ROC-AUC'], 'state': copy.deepcopy(gm.state_dict()), 'step': rnd}
    gm.load_state_dict(best['state'])
    thr = evaluate_model(gm, ctx, 'val')['Threshold']
    return {'test': evaluate_model(gm, ctx, 'test', thr), 'best_step': best['step'], 'history': hist, 'model': gm,
            'uplink_bytes_per_client_round': float(np.mean(uplink)) if uplink else 0.0, 'num_params': sum(p.numel() for p in gm.parameters())}
print("training functions defined")''')

add_code(r'''BASE = 'Popularity baseline (non-learned)'
RESULTS = {'RGCN (Centralized)': [], 'RGCN (Federated)': [], BASE: []}
HISTORY, COMM, STATS_PER_SEED, PRIMARY = {}, [], {}, {}
t_all = time.time()
for seed in SEEDS:
    ctx = ctx0 if seed == SEEDS[0] else prepare(seed)
    STATS_PER_SEED[seed] = ctx['stats']
    t0 = time.time()
    cen, fed = fit_central(ctx, seed), fit_federated(ctx, seed)
    RESULTS['RGCN (Centralized)'].append(cen['test']); RESULTS['RGCN (Federated)'].append(fed['test']); RESULTS[BASE].append(popularity_baseline(ctx))
    COMM.append((fed['num_params'], fed['uplink_bytes_per_client_round']))
    if seed == SEEDS[0]:
        HISTORY = {'central': cen['history'], 'federated': fed['history']}; PRIMARY = {'ctx': ctx, 'fed_model': fed['model'], 'cen_model': cen['model']}
    print(f"seed {seed} | central AUC={cen['test']['ROC-AUC']:.4f} MRR={cen['test']['MRR']:.4f} Hits@10={cen['test']['Hits@10']:.4f} | "
          f"federated AUC={fed['test']['ROC-AUC']:.4f} MRR={fed['test']['MRR']:.4f} Hits@10={fed['test']['Hits@10']:.4f} | "
          f"popularity-only AUC={RESULTS[BASE][-1]['ROC-AUC']:.4f} MRR={RESULTS[BASE][-1]['MRR']:.4f} | {time.time() - t0:.0f}s")
print(f"total time {time.time() - t_all:.0f}s")''')

add_md("# 10. Results (mean ± std over seeds)")
add_code(r'''summary = mean_std_table(RESULTS)
pd.DataFrame([{'model_setting': c, 'seed': s, **r} for c, runs in RESULTS.items() for s, r in zip(SEEDS, runs)]).to_csv('loadhalal_results_per_seed.csv', index=False)
summary.to_csv('loadhalal_results.csv', index=False)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
display(summary)
pool_note = (f"Protocol: {len(SEEDS)} seeds; validation-tuned F1 threshold; 1 positive : {NEG_RATIO} typed negatives (food x manufacturer, filtered; "
             f"majority-class accuracy = {NEG_RATIO / (1 + NEG_RATIO):.3f}); ranking = filtered manufacturer corruption of the tail, {RANKING_CANDIDATES} candidates "
             f"per positive (random baseline MRR ~ {np.mean(1 / np.arange(1, RANKING_CANDIDATES + 2)):.4f}, Hits@10 = {10 / (RANKING_CANDIDATES + 1):.4f}).")
print(pool_note)
pd.DataFrame(HISTORY['federated']).rename(columns={'step': 'round', 'val_auc': 'auc', 'val_f1': 'f1', 'val_prauc': 'prauc', 'val_mrr': 'mrr', 'val_hits10': 'hits10'}).to_csv('loadhalal_federated_history.csv', index=False)''')

add_md(r"""## Reference-paper numbers (indicative only — **not** a like-for-like comparison)
Peeris et al. (*Sustainability* 2026, 18, 2825) Table 4, RGCN on LODHalal: Accuracy 0.7124, AUC-ROC 0.9186, F1 0.1987, MRR 0.0574, Hits@10 0.4315.
Splits, negative sampling and candidate pools differ from this notebook, so only the rough order of magnitude of ROC-AUC is meaningful.""")

add_md(r"""# 11. Ablation: do identity features matter? (first seed)
With `ID_FEATURE_DIM = 0` nodes carry only type and per-relation degree, so a GNN cannot recognise *which* certificate two foods share.
**Result to interpret carefully:** performance with and without identity features is similar, and a *non-learned popularity baseline*
(section 10 table) is also strong. Positives are drawn in proportion to manufacturer size whereas evaluation negatives are uniform
random manufacturers, so size/degree alone separates them. The sampled-negative metrics therefore overstate how well the model
discriminates *between plausible suppliers*; the all-manufacturer ranking in section 13 is the harder, more informative test.""")
add_code(r'''ABL = []
if RUN_ID_ABLATION and ID_FEATURE_DIM > 0:
    ctx_a = prepare(SEEDS[0], id_dim=0)
    ca, fa = fit_central(ctx_a, SEEDS[0]), fit_federated(ctx_a, SEEDS[0])
    for name, r in (('RGCN (Centralized), no ID features', ca['test']), ('RGCN (Federated), no ID features', fa['test'])):
        ABL.append({'setting': name, **{m: r[m] for m in ['ROC-AUC', 'PR-AUC', 'MRR', 'Hits@10', 'Balanced Accuracy']}})
    for name, runs in RESULTS.items():
        r = runs[0]; label = name + ('' if name == BASE else f', ID features (dim {ID_FEATURE_DIM})')
        ABL.append({'setting': label, **{m: r[m] for m in ['ROC-AUC', 'PR-AUC', 'MRR', 'Hits@10', 'Balanced Accuracy']}})
    abl_df = pd.DataFrame(ABL); display(abl_df); abl_df.to_csv('loadhalal_ablation_id_features.csv', index=False)''')

add_md(r"""# 12. Communication (measured serialisation vs theoretical)
Payload sizes of one client's serialised float32 delta in this process — **not** network traffic (no transport/encryption overhead, mask/key exchange, retries or dropouts exist here).""")
add_code(r'''params = COMM[0][0]; measured = float(np.mean([m for _, m in COMM]))
comm_df = pd.DataFrame([{'parameters': params, 'theoretical_payload_MB(4B/param)': params * 4 / 2 ** 20,
                         'measured_serialised_uplink_MB/client/round': measured / 2 ** 20,
                         'estimated_total_uplink_MB (clients x rounds)': measured * NUM_CLIENTS * FED_ROUNDS / 2 ** 20}])
display(comm_df)''')

add_md(r"""# 13. Use case: supplier ranking filtered to certified-halal manufacturers
For each held-out test food we rank **all** manufacturers by model score and then keep only manufacturers that supply at least one certified
food **in the TRAIN edges** (so the filter cannot use held-out labels). When the true manufacturer is excluded by the filter it is
**counted and reported**, not silently skipped.""")
add_code(r'''def supplier_ranking_demo(model, ctx, n_queries=500, seed=0):
    rng = np.random.default_rng(seed)
    model.eval()
    with torch.no_grad():
        ei, et = ctx['eval_graph']
        z = F.normalize(model.encoder(make_x(ctx['static_x'], ei, et, num_nodes), ei, et), dim=-1)
    train = ctx['train_pairs'].numpy()
    halal_set = set(halal_foods.tolist())
    halal_mfr = np.zeros(num_nodes, dtype=bool)
    halal_mfr[train[np.isin(train[:, 0], halal_foods)][:, 1]] = True          # TRAIN edges only
    test = ctx['test']['pos'].numpy()
    test = test[rng.permutation(len(test))[:n_queries]]
    ranks_all, ranks_halal, excluded = [], [], 0
    for f, m in test:
        s = (z[f] * z[MFRS]).sum(-1).numpy()
        t = s[m - MFR_RANGE[0]]
        ranks_all.append(1 + int((s > t).sum()))
        if halal_mfr[m]:
            ranks_halal.append(1 + int((s[halal_mfr[MFRS]] > t).sum()))
        else:
            excluded += 1
    ra, rh = np.array(ranks_all), np.array(ranks_halal)
    print(f"queries={len(test)} | manufacturers={len(MFRS)} (certified-halal-by-train: {int(halal_mfr[MFRS].sum())})")
    print(f"true-manufacturer rank among ALL manufacturers : median={np.median(ra):.0f} mean={ra.mean():.1f} top-10={100*(ra <= 10).mean():.1f}% top-100={100*(ra <= 100).mean():.1f}%")
    if len(rh):
        print(f"rank among certified-halal-by-train only       : median={np.median(rh):.0f} mean={rh.mean():.1f} top-10={100*(rh <= 10).mean():.1f}% (n={len(rh)})")
    print(f"true manufacturer excluded by the halal filter : {excluded}/{len(test)} queries")
    return {'median_rank_all': float(np.median(ra)), 'top10_all': float((ra <= 10).mean()), 'top100_all': float((ra <= 100).mean()),
            'excluded_by_filter': excluded, 'queries': len(test)}

DEMO = {'federated': supplier_ranking_demo(PRIMARY['fed_model'], PRIMARY['ctx'])}
print("(uses only the shared global model and the public certificate relation; random-ranking baseline: median rank ~ 1260 of 2520)")''')

add_md("# 14. Plots")
add_code(r'''os.makedirs('plots', exist_ok=True)
def _despine(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
sm = summary.set_index('Metric')

h = pd.DataFrame(HISTORY['federated'])
fig, ax1 = plt.subplots(figsize=(9, 4.6), dpi=120); ax2 = ax1.twinx()
ax1.plot(h['step'], h['val_auc'], color='#2c7fb8', lw=2.2, marker='o', ms=3.5, label='val ROC-AUC')
ax2.plot(h['step'], h['val_mrr'], color='#d95f0e', lw=2.0, marker='s', ms=3.5, ls='--', label='val MRR')
ax1.set_xlabel('Federated round'); ax1.set_ylabel('Validation ROC-AUC', color='#2c7fb8'); ax2.set_ylabel('Validation MRR', color='#d95f0e'); ax1.grid(alpha=.25, ls='--')
l1, n1 = ax1.get_legend_handles_labels(); l2, n2 = ax2.get_legend_handles_labels(); ax1.legend(l1 + l2, n1 + n2, loc='lower right')
ax1.set_title(f'Federated RGCN (FedAvg, simulated masking, seed {SEEDS[0]})', fontweight='bold')
fig.tight_layout(); fig.savefig('plots/loadhalal_fed_history.png', dpi=150, bbox_inches='tight'); plt.close(fig)

cols = ['RGCN (Centralized)', 'RGCN (Federated)']
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=120)
for ax, metric in zip(axes, ['ROC-AUC', 'PR-AUC', 'MRR']):
    means = [sm.loc[metric, f'{c} mean'] for c in cols]; stds = [sm.loc[metric, f'{c} std'] for c in cols]
    b = ax.bar(cols, means, yerr=stds, capsize=5, color=['#2c7fb8', '#7fcdbb'], edgecolor='white'); ax.bar_label(b, fmt='%.3f', padding=3)
    ax.set_title(f'{metric} (mean ± std, {len(SEEDS)} seeds)', fontweight='bold'); ax.grid(axis='y', alpha=.3, ls='--'); _despine(ax)
    if metric == 'ROC-AUC': ax.axhline(0.5, color='grey', ls='--', lw=.8)
    if metric == 'MRR': ax.axhline(np.mean(1 / np.arange(1, RANKING_CANDIDATES + 2)), color='grey', ls='--', lw=.8)
fig.tight_layout(); fig.savefig('plots/loadhalal_model_comparison.png', dpi=150, bbox_inches='tight'); plt.close(fig)

s0 = ctx0['stats']; cc = ['#2c7fb8', '#41b6c4', '#7fcdbb', '#d9ef8b']
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.4), dpi=120)
a1.bar(range(1, NUM_CLIENTS + 1), s0['nodes_per_client'], color=cc, edgecolor='white'); a1.set_title('Nodes owned per client', fontweight='bold')
a2.bar(range(1, NUM_CLIENTS + 1), s0['local_target_edges_per_client'], color=cc, edgecolor='white'); a2.set_title('Local train target edges', fontweight='bold')
keys = ['cross_client_train_target', 'cross_client_val_target', 'cross_client_test_target', 'cross_client_contains', 'cross_client_halal_status']
a3.bar(['train\ntarget', 'val\ntarget', 'test\ntarget', 'contains', 'halal\nstatus'], [s0[k] for k in keys], color='#d95f0e', edgecolor='white')
a3.set_ylim(0, 1); a3.set_title('Cross-client share of edges', fontweight='bold')
for ax in (a1, a2, a3): ax.grid(axis='y', alpha=.3, ls='--'); _despine(ax)
fig.tight_layout(); fig.savefig('plots/loadhalal_client_distribution.png', dpi=150, bbox_inches='tight'); plt.close(fig)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120)
types = list(RANGES); a1.bar(types, [RANGES[t][1] - RANGES[t][0] for t in types], color='#2c7fb8', edgecolor='white'); a1.set_title('Nodes per type', fontweight='bold')
a2.bar(['has_manufacturer', 'contains', 'halal_status'], [q['kept'] for q in EDGE_QUALITY], color='#d95f0e', edgecolor='white', label='kept')
a2.bar(['has_manufacturer', 'contains', 'halal_status'], [q['dropped_missing_endpoint'] for q in EDGE_QUALITY], bottom=[q['kept'] for q in EDGE_QUALITY], color='#bbbbbb', edgecolor='white', label='dropped (missing endpoint)')
a2.set_title('Edges per relation (kept vs dropped)', fontweight='bold'); a2.legend(fontsize=8)
for ax in (a1, a2): ax.grid(axis='y', alpha=.3, ls='--'); _despine(ax)
fig.tight_layout(); fig.savefig('plots/loadhalal_dataset_structure.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print("plots written")''')

add_md("# 15. Run metadata (reproducibility record)")
add_code(r'''meta = {
    'notebook': 'loadhalal_federated_kg_completion.ipynb',
    'run_timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
    'seeds': SEEDS,
    'config': {k: globals()[k] for k in ['HIDDEN_DIM', 'DROPOUT', 'LEARNING_RATE', 'NUM_BASES', 'SUP_FRAC', 'ID_FEATURE_DIM', 'NUM_CLIENTS', 'LOCAL_EPOCHS',
                                        'FED_ROUNDS', 'CENTRAL_EPOCHS', 'NEG_RATIO', 'RANKING_CANDIDATES', 'SPLIT', 'PAIRWISE_MASK_SIM', 'DP_SIGMA', 'DP_CLIP_NORM']},
    'privacy_status': {'differential_privacy': 'OFF (DP_SIGMA=0): no (epsilon, delta) guarantee',
                       'secure_aggregation': 'simulated pairwise masking in one process; not a deployable protocol'},
    'evaluation_protocol': pool_note,
    'edge_quality': EDGE_QUALITY,
    'manufacturer_parse': {'rows': int(len(manufacturer_df)), 'names_with_commas': int(n_comma_names), 'names_truncated_by_usecols': truncated},
    'per_seed_split_stats': {str(k): v for k, v in STATS_PER_SEED.items()},
    'supplier_ranking_demo': DEMO,
    'leakage_acceptance_tests': 'passed in prepare() for every seed',
}
with open('run_metadata_loadhalal.json', 'w') as f:
    json.dump(meta, f, indent=2, default=str)
print("metadata written")''')

add_md(r"""# 16. Interpretation and Claims

## What the results can support
- Centralized vs simulated-federated RGCN for `has_manufacturer` prediction under a leakage-free protocol (mean ± std over seeds).
- Much of the apparent performance is explained by **manufacturer popularity**: a non-learned baseline that scores a manufacturer by how many
  training foods it has is also strong under the sampled-negative protocol (section 10), and removing identity features barely changes the
  GNN's numbers (section 11). In the reported run the RGCN is clearly better than that baseline on ROC-AUC / PR-AUC / balanced accuracy, but on
  sampled-candidate MRR and Hits@10 it is close to it. Compare against that baseline, not against the random-guess line.
- Certificates are almost manufacturer-specific in this data (about 1.04 manufacturers per certificate), so a held-out food *could* be linked to its
  manufacturer through a shared certificate with training foods. Using the `halal_status` relation this way is legitimate (it is not target-edge leakage), but
  the all-manufacturer ranking in section 13 shows the model only partly exploits it.

## What the results do **not** support
- **No privacy guarantee** (DP off; masking is a one-process arithmetic simulation, not Secure Aggregation).
- **No deployment claim** (simulated clients in one process).
- **No like-for-like comparison with the reference paper** (different splits, negatives, candidate pools).
- Ranking metrics depend on the 50-candidate pool; thresholded metrics depend on the validation-tuned threshold and 1:5 class balance.

## Known limitations
- ~98% of `food_ingredient` rows reference food ids that are absent from `food.csv`; the `contains` relation is therefore tiny (data issue, reported in section 4).
- Each food has exactly one manufacturer, so train target edges are 100% intra-client by construction of the ownership rule; cross-client signal is lost
  for the non-target relations and for validation/test foods.
- Clients draw negatives only from their own manufacturers, while evaluation ranks all manufacturers (a real federated distribution shift).
- Communication numbers are payload sizes, not network measurements.

## Future work (not implemented)
Real Secure Aggregation in an FL framework; calibrated DP with more clients; multi-owner food ownership; measured multi-process communication.""")

n_cells = nb.write("loadhalal_federated_kg_completion.ipynb")
print(f"Notebook written: loadhalal_federated_kg_completion.ipynb ({n_cells} cells)")
