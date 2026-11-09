"""Relational pointers with a local clamped/free-phase Adagrad rule."""
import argparse, json, math, time
import torch
import iter10 as R
from hybrid_mixer import Neuron
I = R.I
torch.set_grad_enabled(False)
RULES = ['Reference iter10 clock: test=1.1711, renamed=1.6269, gap=0.4559.', 'Backprop 16M: test=1.1238, renamed=1.5298, gap=0.406.', 'learned must beat frozen by >=0.01 renamed, else local rule is not learning.', 'learned renamed<=1.58 and gap<0.43: name dependence improved; continue.', 'learned test<=1.16: continue.', 'pure: report only.']

def rows(x, word):
    b, T = x.shape; pos = torch.arange(T, device=x.device); valid = pos[None, :] < pos[:, None]
    dist = torch.bucketize((pos[:, None]-pos[None, :]).contiguous(), pos.new_tensor([2,4,8,16,32,64])); w = word[x]
    h = torch.zeros_like(x); p1 = h.clone(); p2 = h.clone(); length = h.clone()
    for t in range(T):
        cont = w[:, t] & w[:, t-1] if t else torch.zeros(b, dtype=torch.bool, device=x.device)
        if t: p1[:, t] = torch.where(cont, p1[:, t-1], h[:, t-1]); p2[:, t] = torch.where(cont, p2[:, t-1], p1[:, t-1])
        h[:, t] = torch.where(cont, h[:, t-1] if t else 0, 0)*1000003+x[:, t]+1
        length[:, t] = torch.where(cont, length[:, t-1] if t else 0, 0)+1
    length.clamp_max_(7); nxt = torch.cat([x[:, 1:], torch.full_like(x[:, :1], len(word)-1)], 1)
    out = [torch.empty((b,T,T), dtype=torch.int16, device=x.device) for _ in range(2)]
    for a in range(0,b,32):
        z = x[a:a+32]; bits = torch.zeros((len(z),T,T), dtype=torch.int16, device=x.device)
        for lag in range(8):
            ix = (pos-lag).clamp_min(0); eq = (z[:, ix, None] == z[:, None, ix]) & (pos[:, None]>=lag) & (pos[None, :]>=lag); bits.add_(eq.to(torch.int16)*(1<<lag))
        out[0][a:a+32] = torch.where(valid, bits*7+dist, -1).short(); bits.zero_()
        features = [(v[a:a+32, :, None] == v[a:a+32, None, :]) for v in (h,p1,p2,length)]
        features += [word[nxt[a:a+32, None, :]].expand(-1,T,-1), w[a:a+32, :, None].expand(-1,-1,T)]
        for k, eq in enumerate(features): bits.add_(eq.to(torch.int16)*(1<<k))
        out[1][a:a+32] = torch.where(valid, bits*7+dist, -1).short()
    return out+[nxt[:, None, :].expand(-1,T,-1)]

class Head:
    def __init__(self, n, device):
        self.W = torch.full((n,), -5., device=device); self.w0 = torch.zeros(1, device=device); self.G = torch.full((n,), 1e-8, device=device); self.G0 = torch.full((1,), 1e-8, device=device)
    def __call__(self, row, cand, f, y=None):
        f = f.clamp_min(1e-6); f = f/f.sum(-1, keepdim=True); ix = row.long().clamp_min(0)
        a = torch.cat([self.w0.expand(len(f),1), self.W[ix].masked_fill(row<0, -torch.inf)], 1).softmax(-1)
        proposals = f.new_zeros((*cand.shape,f.shape[1]+1)).scatter_(2,cand[:,:,None],1)
        q = a[:, :1]*f+torch.bmm(a[:,None,1:],proposals[:,:,:-1])[:,0]; q = q/q.sum(-1, keepdim=True); delta = None
        if y is not None:
            qy = q.gather(1,y[:,None]); d = a[:,1:]*((cand == y[:,None])/qy-1); delta = torch.zeros_like(self.W).scatter_add_(0,ix.reshape(-1),d.reshape(-1)); d0 = (a[:,:1]*(f.gather(1,y[:,None])/qy-1)).sum().reshape(1)
            for W,G,D in ((self.W,self.G,delta),(self.w0,self.G0,d0)):
                use = D != 0; G[use] += D[use].square(); W[use] += .2*D[use]/G[use].sqrt()
        return q, a, delta

class Model(R.Model):
    def __init__(self, V, variant='learned', lr=1e-4):
        self.mode = variant
        if variant != 'pure': super().__init__(V, lr, variant='clock')
        self.V, self.name, self.lr = V, 'mix_two_layer', lr
        vocab = json.loads((I.ROOT/'data/meta.json').read_text())['vocab']; self.word = torch.tensor([c.isalnum() or c=='_' for c in vocab]+[False], device='cuda')
        sizes = (1,V+1,16) if variant=='pure' else (1,V+1,4096,16,12,16); self.first = [Neuron(r,3 if variant=='pure' else 18,lr) for r in sizes]; self.second = Neuron(V+1,len(sizes),lr)
        for n in self.first+[self.second]: n.w = n.w.cuda()
        self.heads = [Head(n,'cuda') for n in (1792,448)]
    def prepare(self, x):
        if self.mode == 'pure':
            f = torch.nn.functional.one_hot(x,self.V+1)[...,:self.V].float().cumsum(1)+.5; cod, memory, sel = {}, [f/f.sum(-1,keepdim=True)], self.selectors(x)[:2]
        else: cod, memory, sel = super().prepare(x)
        return cod, memory+rows(x,self.word), sel
    def output(self, cod, selectors, memory, y=None, nulls=None):
        rel = (*memory[-3:],nulls)
        if self.mode == 'pure': return self.mix([memory[0]],selectors+[rel,None],y)
        return super().output(cod,selectors[:-1]+[rel]+selectors[-1:],memory[:-3],y)
    def mix(self, ps, selectors, y=None):
        r1,r2,cand,nulls = selectors[-2]; sel = selectors[:-2] if self.mode=='pure' else selectors[:-2]+selectors[-1:]
        qa = [h(r,cand,ps[0] if self.mode=='pure' else ps[4],y if self.mode!='frozen' else None) for h,r in zip(self.heads,(r1,r2))]
        bucket = [torch.bucketize((1-a[:,0]).contiguous(),a.new_tensor([.1,.5,.9])) for _,a,_ in qa]
        if nulls is not None: nulls.add_(torch.stack([qa[0][1][:,0].sum(),qa[1][1][:,0].sum(),qa[0][1].new_tensor(len(cand))]))
        return I.Model.mix(self,ps+[q for q,_,_ in qa],sel+[bucket[0]*4+bucket[1]],y)
    def logprobs(self, x, nulls=None):
        cod, memory, sel = self.prepare(x); T = x.shape[1]; cod = {n:c.reshape(-1) for n,c in cod.items()}; sel = [s.reshape(-1) for s in sel]
        memory = [c.reshape(x.numel(),-1) if c.ndim==3 else c.reshape(-1) for c in memory]
        return torch.cat([self.output({n:c[a:a+1024] for n,c in cod.items()},[s[a:a+1024] for s in sel],[c[a:a+1024] for c in memory],nulls=nulls) for a in range(0,x.numel(),1024)]).reshape(*x.shape,self.V)

def reference(vocab):
    V = len(vocab); word = torch.tensor([c.isalnum() or c=='_' for c in vocab]+[False]); x = torch.tensor([[V]+[vocab.index(c) for c in s] for s in ('ab_2 ab_2+ab_2 A!','ab_2+ab_2 ab_2 A!')]); rr = rows(x,word)
    chars = [i for i,c in enumerate(vocab) if c.isalpha()]; perm = torch.arange(V+1); perm[chars] = torch.tensor(chars)[torch.randperm(len(chars))]; assert all(torch.equal(a,b) for a,b in zip(rr[:2],rows(perm[x],word)[:2])), 'renaming changed relations'
    for b, seq in enumerate(x.tolist()):
        tokens = []; features = []; current = ()
        for t,c in enumerate(seq):
            if t and word[c] and word[seq[t-1]]: current += (c,)
            else:
                if t: tokens.append(current)
                current = (c,)
            features.append((current,tokens[-1] if tokens else (),tokens[-2] if len(tokens)>1 else (),min(len(current),7)))
        for t in range(len(seq)):
            for j in range(len(seq)):
                d = sum(t-j>v for v in (2,4,8,16,32,64)); bit1 = sum((t>=l and j>=l and seq[t-l]==seq[j-l])<<l for l in range(8)); bit2 = sum(int(features[t][k]==features[j][k])<<k for k in range(4))+(int(word[seq[j+1]])<<4 if j+1<len(seq) else 0)+(int(word[seq[t]])<<5)
                assert int(rr[0][b,t,j]) == (bit1*7+d if j<t else -1) and int(rr[1][b,t,j]) == (bit2*7+d if j<t else -1), 'row reference'
    for k,n in enumerate((1792,448)):
        h = Head(n,'cpu'); h.W.copy_(torch.linspace(-5,-1,n)); h.w0.fill_(.3); r = rr[k][:,12]; cand = rr[2][:,12]; f = torch.arange(1,V+1).float().expand(len(x),-1); f = f/f.sum(-1,keepdim=True); y = x[:,5]; W = h.W.clone(); D = torch.zeros_like(W); D0 = 0.; aa = []; qq = []
        for b in range(len(x)):
            scores = [math.exp(float(h.w0[0]))]+[math.exp(float(W[int(r[b,j])])) if int(r[b,j])>=0 else 0. for j in range(x.shape[1])]; a = [s/sum(scores) for s in scores]; q = [a[0]*float(f[b,c])+sum(a[j+1] for j in range(x.shape[1]) if int(cand[b,j])==c) for c in range(V)]; aa.append(a); qq.append(q)
            for j in range(x.shape[1]):
                if int(r[b,j])>=0: D[int(r[b,j])] += a[j+1]*(int(cand[b,j])==int(y[b]))/q[int(y[b])]-a[j+1]
            D0 += a[0]*(float(f[b,y[b]])/q[int(y[b])]-1)
        q,a,delta = h(r,cand,f,y); torch.testing.assert_close(a,torch.tensor(aa)); torch.testing.assert_close(q,torch.tensor(qq)); torch.testing.assert_close(delta,D,atol=1e-7,rtol=1e-5)
        torch.testing.assert_close(h.G,1e-8+D.square()); torch.testing.assert_close(h.W,W+.2*D/(1e-8+D.square()).sqrt(),atol=2e-5,rtol=1e-5); torch.testing.assert_close(h.G0,torch.tensor([1e-8+D0**2])); torch.testing.assert_close(h.w0,torch.tensor([.3+.2*D0/math.sqrt(1e-8+D0**2)]))

def run(name, budget=None, smoke=False):
    if budget is not None and budget<128: raise ValueError('budget must be >=128')
    if smoke and (budget is None or budget>256): raise ValueError('smoke requires budget <=256')
    torch.manual_seed(0); reference(json.loads((I.ROOT/'data/meta.json').read_text())['vocab']); torch.cuda.set_per_process_memory_fraction(2.7*2**30/torch.cuda.get_device_properties(0).total_memory); torch.cuda.reset_peak_memory_stats()
    tr,te,V = I.B.load(); cut = int(len(tr)*.95); m = Model(V,name); R.reference() if name=='learned' else None; torch.cuda.synchronize(); start = time.perf_counter(); chars = 0
    for x,y in I.batches(tr[:cut],budget,976,True,V):
        cod, memory, sel = m.prepare(x)
        for t in range(x.shape[1]):
            keep = y[:,t]>=0; target = y[:,t][keep]
            if not len(target): continue
            c = {n:v[:,t][keep] for n,v in cod.items()}; m.output(c,[s[:,t][keep] for s in sel],[v[:,t][keep] for v in memory],target)
            if name!='pure': m.base.write(c,target)
        chars += int((y>=0).sum())
    torch.cuda.synchronize(); seconds = time.perf_counter()-start; before = I.stamp(m); renamed,info = I.rename(te); nulls = torch.zeros(3,device='cuda'); old_windows = I.B.N_EVAL_WINDOWS
    try:
        if smoke: I.B.N_EVAL_WINDOWS = 2
        test = I.evaluate(lambda x:m.logprobs(x,nulls),te); ren = I.evaluate(m.logprobs,renamed); val = I.evaluate(m.logprobs,tr[cut:]); R.R.causality(m,te); assert I.stamp(m)==before, 'evaluation wrote state'
    finally: I.B.N_EVAL_WINDOWS = old_windows
    settings = dict(seed=0,device='cuda',bs=976,T=128,lr=1e-4,budget=budget,sampling='one_pass',train_end=cut,val_start=cut,allocator_limit_gib=2.7,variant=name,clock_variant=None if name=='pure' else 'clock',inputs=3 if name=='pure' else 18,first_rows=[n.w.shape[0] for n in m.first],second_inputs=len(m.first),head_rows=[1792,448],head_lr=.2,head_init=-5.,null_init=0.,adagrad_init=1e-8,batch_delta='sum; per position',probability_floor=1e-6,fallback='window histogram +0.5 excluding pad' if name=='pure' else 'ps[4]',frozen=name=='frozen',evaluation_frozen=True,eval_chunk=1024,eval_windows=2 if smoke else old_windows,renaming=info,decision_rules=RULES,smoke=smoke)
    result = dict(test_loss=test,renamed_loss=ren,val_loss=val,renamed_gap=ren-test,train_seconds=seconds,chars=chars,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,head1_null_mean=float(nulls[0]/nulls[2]),head2_null_mean=float(nulls[1]/nulls[2]),settings=settings)
    with (I.ROOT/f'results/iter12_{"smoke_" if smoke else ""}{name}.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
    return result
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('name',choices=('learned','frozen','pure')); p.add_argument('--budget',type=int); p.add_argument('--smoke',action='store_true'); a = p.parse_args(); print(json.dumps(run(a.name,a.budget,a.smoke),indent=2),flush=True)
