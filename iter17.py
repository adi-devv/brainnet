"""iter15 `local`, unchanged, run for several shuffled passes over the training data."""
import json, math, sys, time
import numpy as np
import torch
import iter15 as Q
I = Q.I; B = I.B

def epoch(tr, seed, left, V, bs=16):
    """Same partition as iter5.batches(one_pass=True); the seed gives each pass its own order."""
    g = torch.Generator().manual_seed(seed); data = torch.as_tensor(tr.astype(np.int64), device='cuda')
    pos = torch.arange(128, device='cuda'); starts = torch.randperm(math.ceil(len(tr)/128), generator=g)*128
    for a in range(0, len(starts), bs):
        st = starts[a:a+min(bs, math.ceil(left/128))]
        if not len(st): break
        ix = st.cuda()[:, None]+pos
        valid = (ix < len(tr)) & (torch.arange(ix.numel(), device='cuda').reshape_as(ix) < left)
        yield torch.where(ix > 0, data[(ix-1).clamp(0, len(tr)-1)], V), torch.where(valid, data[ix.clamp_max(len(tr)-1)], -1)
        left -= int(valid.sum())
        if left <= 0: break

def run(passes, cosine=False):
    path = I.ROOT/f'results/iter17_local_{passes}p{"_cosine" if cosine else ""}.json'
    if path.exists(): raise FileExistsError(path)
    assert torch.cuda.is_available() and not torch.is_grad_enabled()
    torch.manual_seed(0); torch.cuda.reset_peak_memory_stats()
    tr, te, V = B.load(); cut = int(len(tr)*.95); train = tr[:cut]; renamed, info = I.rename(te)
    net = Q.Model(V, False)
    def predict(x, layer=None):
        zs = net.forward(x)[1]
        return (zs[layer] if layer is not None else sum(w*z for w, z in zip(net.p['g'], zs))).log_softmax(-1)
    seconds = 0.; chars = 0; curve = []; top = net.lr; total = math.ceil(passes*len(train)/(16*128))
    for p in range(math.ceil(passes)):
        left = int(len(train)*min(1, passes-p))
        torch.cuda.synchronize(); start = time.perf_counter()
        for x, y in epoch(train, p, left, V):
            if cosine: net.lr = top*(.1+.9*.5*(1+math.cos(math.pi*min(1, net.step/total))))
            net.train(x, y, 0); chars += int((y >= 0).sum())
        torch.cuda.synchronize(); seconds += time.perf_counter()-start
        test = I.evaluate(predict, te); ren = I.evaluate(predict, renamed)
        curve.append(dict(passes=min(p+1, passes), chars=chars, train_seconds=seconds, test_loss=test, renamed_loss=ren))
        print(json.dumps(curve[-1]), flush=True)
    result = Q.novel(predict, te, renamed, json.loads((I.ROOT/'data/meta.json').read_text())['vocab'])
    result.update(curve=curve, test_loss=test, renamed_loss=ren, renamed_gap=ren-test, val_loss=I.evaluate(predict, tr[cut:]),
                  per_layer_test_loss=[I.evaluate(lambda x, l=l: predict(x, l), te) for l in range(net.L)], train_seconds=seconds, chars=chars,
                  steps=net.step, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, passes=passes, renaming=info,
                  settings=dict(seed=0, bs=16, T=128, D=256, layers=3, lr=1e-3, lr_schedule='cosine to lr/10' if cosine else 'constant', sampling='shuffled passes, seed=pass index'))
    with path.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ('curve', 'renaming', 'settings')}), flush=True)

if __name__ == '__main__': run(float(sys.argv[1]), 'cosine' in sys.argv[2:])
