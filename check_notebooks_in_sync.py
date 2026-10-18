"""Verifies that the committed notebooks' SOURCE CELLS match what the generators produce (audit item D3).

    python check_notebooks_in_sync.py          # exit code 0 = in sync, 1 = out of sync

It regenerates both notebooks into a temporary directory (so nothing in the project is overwritten)
and compares the code/markdown sources cell by cell. Outputs are ignored.
"""
import json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = [("generate_notebook.py", "anonnet_federated_kg_completion.ipynb"),
         ("generate_loadhalal.py", "loadhalal_federated_kg_completion.ipynb")]

def sources(path):
    with open(path, encoding="utf-8") as f:
        return [(c["cell_type"], "".join(c["source"])) for c in json.load(f)["cells"]]

ok = True
with tempfile.TemporaryDirectory() as tmp:
    for name in ["notebook_common.py"] + [g for g, _ in PAIRS]:
        shutil.copy(os.path.join(HERE, name), tmp)
    for gen, nbname in PAIRS:
        subprocess.run([sys.executable, gen], cwd=tmp, check=True, stdout=subprocess.DEVNULL)
        fresh, committed = sources(os.path.join(tmp, nbname)), sources(os.path.join(HERE, nbname))
        if fresh == committed:
            print(f"OK   {nbname} matches {gen} ({len(fresh)} cells)")
        else:
            ok = False
            diff = [i for i, (a, b) in enumerate(zip(fresh, committed)) if a != b]
            print(f"DIFF {nbname} differs from {gen}: cell count {len(committed)} vs {len(fresh)}, first differing cells {diff[:5]}")
sys.exit(0 if ok else 1)
