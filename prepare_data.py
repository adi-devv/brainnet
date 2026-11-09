import os, json, hashlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.__file__)
files = []
for dp, dn, fn in os.walk(root):
    dn[:] = sorted(d for d in dn if d not in ("test","tests","idlelib","site-packages","__pycache__"))
    for f in sorted(fn):
        if f.endswith(".py"):
            files.append(os.path.join(dp, f))
files.sort()
keep = set(chr(i) for i in range(32,127)) | {"\n","\t"}
CAP = 8_000_000
texts = []; total = 0
for p in files:
    t = open(p, encoding="utf-8", errors="ignore").read().replace("\r\n","\n")
    t = "".join(c for c in t if c in keep)
    if not t: continue
    if total + len(t) > CAP: continue
    texts.append((os.path.relpath(p, root), t)); total += len(t)
# split by file: every 10th file (by sorted index) -> test
train = "".join(t for i,(n,t) in enumerate(texts) if i % 10 != 9)
test = "".join(t for i,(n,t) in enumerate(texts) if i % 10 == 9)
vocab = sorted(set(train+test))
stoi = {c:i for i,c in enumerate(vocab)}
tr = np.array([stoi[c] for c in train], dtype=np.uint8)
te = np.array([stoi[c] for c in test], dtype=np.uint8)
os.makedirs(HERE+"/data", exist_ok=True)
tr.tofile(HERE+"/data/train.bin"); te.tofile(HERE+"/data/test.bin")
counts = {c: int((tr==stoi[c]).sum()) for c in vocab}
meta = dict(vocab=vocab, train_counts=counts, n_files=len(texts),
  n_train_files=sum(1 for i in range(len(texts)) if i%10!=9), n_test_files=sum(1 for i in range(len(texts)) if i%10==9),
  train_chars=len(tr), test_chars=len(te),
  train_sha256=hashlib.sha256(tr.tobytes()).hexdigest(), test_sha256=hashlib.sha256(te.tobytes()).hexdigest(),
  source_root=root)
json.dump(meta, open(HERE+"/data/meta.json","w"), indent=1)
print({k:v for k,v in meta.items() if k not in("vocab","train_counts")}, len(vocab))
