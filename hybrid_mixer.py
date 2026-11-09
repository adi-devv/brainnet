"""Iteration 4: local logistic mixers; all predictions precede memory writes."""
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, math, time
import numpy as np
import torch
import benchmark as B
from hybrid import Hybrid, LENS

torch.set_grad_enabled(False)
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
NAMES = ("hedge_recheck", "mix_global", "mix_char", "mix_two_layer")
BUDGET = 122 * 64 * 128


def state_digest(obj):
    h = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            h.update(str((value.dtype, tuple(value.shape))).encode())
            h.update(memoryview(value.contiguous().numpy()).cast("B"))
        elif isinstance(value, dict):
            for key, val in value.items():
                visit(key); visit(val)
        elif isinstance(value, (list, tuple)):
            for val in value: visit(val)
        elif hasattr(value, "__dict__"): visit(vars(value))
        else: h.update(repr(value).encode() + b"\0")
    visit(obj)
    return h.hexdigest()


class Neuron:
    def __init__(self, rows, inputs, lr):
        self.w = torch.full((rows, inputs), 1.0 / inputs)
        self.lr = lr

    def predict(self, lp, selector):
        return (self.w[selector, :, None] * lp).sum(1).log_softmax(-1)

    def learn(self, lp, selector, logq, y):
        observed = lp[torch.arange(len(y)), :, y]
        expected = (logq.exp()[:, None, :] * lp).sum(-1)
        # Each example contributes its own delta; repeated selector rows add.
        self.w.index_add_(0, selector, self.lr * (observed - expected))


class Mixer:
    def __init__(self, V, name, lr=0.02):
        if name not in NAMES: raise ValueError(name)
        self.base = Hybrid(V, mode="ngram", surprise=False)
        self.V, self.name, self.lr = V, name, lr
        rows = {"hedge_recheck": [], "mix_global": [1], "mix_char": [V + 1],
                "mix_two_layer": [1, V + 1, 4096]}[name]
        self.first = [Neuron(r, len(LENS), lr) for r in rows]
        self.second = Neuron(V + 1, 3, lr) if name == "mix_two_layer" else None

    def selectors(self, x):
        # At t, last char is x[t]; at t=0 its predecessor is pad=V.
        prev = torch.cat([torch.full_like(x[:, :1], self.V), x[:, :-1]], 1)
        return [torch.zeros_like(x), x, (prev * 1000003 + x) % 4096]

    def mix(self, ps, selectors, y=None):
        lp = torch.stack(ps, 1).clamp_min(1e-6).log()
        chosen = selectors if self.second else [selectors[1 if self.name == "mix_char" else 0]]
        outputs = [n.predict(lp, c) for n, c in zip(self.first, chosen)]
        result = outputs[0]
        if self.second:
            upper = torch.stack(outputs, 1).clamp_min(math.log(1e-6))
            result = self.second.predict(upper, selectors[1])
        if y is not None:
            for n, c, q in zip(self.first, chosen, outputs): n.learn(lp, c, q, y)
            if self.second: self.second.learn(upper, selectors[1], result, y)
        return result

    def train(self, tr, budget=BUDGET, seed=0, bs=64, T=128):
        if self.name == "hedge_recheck":
            return self.base.train(tr, budget, seed, bs, T)
        g = torch.Generator().manual_seed(seed)
        trt, ar = torch.from_numpy(tr.astype(np.int64)), torch.arange(T)
        for _ in range(budget // (bs * T)):
            st = torch.randint(0, len(tr) - T - 1, (bs,), generator=g)
            x, y = trt[st[:, None] + ar], trt[st[:, None] + ar + 1]
            cods, selectors = self.base._codes(x), self.selectors(x)
            for t in range(T):
                cod = {n: c[:, t] for n, c in cods.items()}
                _, ps, _ = self.base._predict(cod, False)
                self.mix(ps, [c[:, t] for c in selectors], y[:, t])
                self.base.step(cod, y[:, t])

    def logprobs(self, x, chunk=4096):
        if self.name == "hedge_recheck": return self.base.logprobs(x, chunk)
        bn, T = x.shape
        cods = {n: c.reshape(-1) for n, c in self.base._codes(x).items()}
        selectors = [c.reshape(-1) for c in self.selectors(x)]
        out = []
        for i in range(0, bn * T, chunk):
            _, ps, _ = self.base._predict({n: c[i:i + chunk] for n, c in cods.items()}, False)
            out.append(self.mix(ps, [c[i:i + chunk] for c in selectors]))
        return torch.cat(out).reshape(bn, T, self.V)


def run(name, lr=0.02):
    torch.manual_seed(0)
    tr, _, V = B.load()
    model = Mixer(V, name, lr)
    start = time.perf_counter()
    model.train(tr)
    seconds = time.perf_counter() - start
    before = state_digest(model)
    loss = B.evaluate(model.logprobs, device="cpu")
    assert state_digest(model) == before, "model changed during evaluation"
    if name == "hedge_recheck": assert abs(loss - 1.6727) <= 0.002, loss
    result = dict(name=name, test_loss=loss, train_seconds=seconds,
                  chars_per_sec=BUDGET / seconds,
                  settings=dict(seed=0, batches=122, bs=64, T=128, budget=BUDGET,
                                threads=2, interop_threads=1, mode="ngram", surprise=False,
                                lens=list(LENS), lr=lr, initial_weight="1/n_inputs",
                                probability_floor=1e-6, selector_pad=V, hash_rows=4096,
                                selector_hash="(previous * 1000003 + current) % 4096",
                                batch_delta="sum", local_losses=True,
                                kill_rule="mix_two_layer must beat hedge_recheck by >= 0.1"))
    with open(f"{B.HERE}/results/iter4_{name}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=NAMES)
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()
    print(json.dumps(run(args.variant, args.lr), indent=2), flush=True)
