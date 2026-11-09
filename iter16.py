"""Locked backprop baseline scored on the renamed-name and novel-name tests at a given char budget."""
import json, sys
import torch
import iter15 as Q
import baseline
I = Q.I; B = I.B

def run(budget):
    path = I.ROOT/f'results/iter16_backprop_{budget}.json'
    if path.exists(): raise FileExistsError(path)
    _, te, _ = B.load(); renamed, info = I.rename(te)
    original = B.evaluate; captured = []
    def capture(fn, *args, **kwargs):
        captured[:] = [fn]; return original(fn, *args, **kwargs)
    B.evaluate = capture
    try:
        with torch.enable_grad(): result = baseline.train(0, budget, verbose=False)
    finally: B.evaluate = original; torch.set_grad_enabled(False)
    fn = captured[0]; test = I.evaluate(fn, te); ren = I.evaluate(fn, renamed)
    vocab = json.loads((I.ROOT/'data/meta.json').read_text())['vocab']
    result.update(Q.novel(lambda x: fn(x).float(), te, renamed, vocab))
    result.update(test_loss_recheck=test, renamed_loss=ren, renamed_gap=ren-test, budget=budget, renaming=info)
    with path.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ('curve', 'renaming')}), flush=True)

if __name__ == '__main__':
    for b in sys.argv[1:]: run(int(b))
