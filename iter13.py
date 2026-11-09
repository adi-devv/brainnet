"""Character-kind relational pointer head with Iter 12's local Adagrad rule."""
import argparse, json
import torch
import iter12 as R
from hybrid_mixer import Neuron
I = R.I
torch.set_grad_enabled(False)
RULES = ['Reference iter12 learned: test=1.1651, renamed=1.6058, gap=0.4407.', 'Backprop at 16M chars: test=1.1238, renamed=1.5298, gap=0.406.', 'Beats iter12 learned by >=0.01 renamed loss: the class head adds something.', 'Renamed<=1.58 and gap<0.43: name dependence improved; continue this learner.', 'Test<=1.16: continue.', 'If neither of the last two conditions holds: stop stacking on count tables.']

def keys(vocab, name):
    out = []; K = 2
    for c in list(vocab)+[None]:
        if c is not None and c.isalpha(): key = 0
        elif c is not None and c.isdigit(): key = 1
        elif name == 'classes': key = {'_':2, ' ':3, '\n':4}.get(c,5)
        else: key = K; K += 1
        out.append(key)
    return torch.tensor(out, dtype=torch.int32), 6 if name == 'classes' else K

def rows3(x, key, K, rr):
    # Head 1's low two bits are eq0/eq1; head 2's low bit is h equality.
    r1,r2,cand = rr; out = torch.empty_like(r1, dtype=torch.int32)
    for a in range(0,len(x),32):
        u = r1[a:a+32].int(); v = r2[a:a+32].int()
        kind = key[x[a:a+32, :, None]]*K+key[cand[a:a+32]]
        bits = ((u//7)&3)+4*((v//7)&1)
        out[a:a+32] = torch.where(u>=0, (kind*8+bits)*7+u%7, -1)
    return out

class Model(R.Model):
    def __init__(self, V, name, lr=1e-4):
        super().__init__(V, 'learned', lr)
        vocab = json.loads((I.ROOT/'data/meta.json').read_text())['vocab']; key,self.K = keys(vocab,name); self.key = key.cuda()
        self.first = [Neuron(r,19,lr) for r in (1,V+1,4096,16,12,16)]; self.second = Neuron(V+1,6,lr)
        for n in self.first+[self.second]: n.w = n.w.cuda()
        self.heads.append(R.Head(self.K*self.K*8*7,'cuda'))
    def prepare(self, x):
        cod,memory,sel = super().prepare(x)
        return cod,memory[:-1]+[rows3(x,self.key,self.K,memory[-3:]),memory[-1]],sel
    def output(self, cod, selectors, memory, y=None, nulls=None):
        rel = (*memory[-4:],nulls)
        return R.R.Model.output(self,cod,selectors[:-1]+[rel]+selectors[-1:],memory[:-4],y)
    def mix(self, ps, selectors, y=None):
        r1,r2,r3,cand,nulls = selectors[-2]; sel = selectors[:-2]+selectors[-1:]
        qa = [h(r,cand,ps[4],y) for h,r in zip(self.heads,(r1,r2,r3))]
        bucket = [torch.bucketize((1-a[:,0]).contiguous(),a.new_tensor([.1,.5,.9])) for _,a,_ in qa[:2]]
        if nulls is not None: nulls.add_(torch.stack([a[:,0].sum() for _,a,_ in qa]+[qa[0][1].new_tensor(len(cand))]))
        return I.Model.mix(self,ps+[q for q,_,_ in qa],sel+[bucket[0]*4+bucket[1]],y)

def reference(vocab):
    V = len(vocab); word = torch.tensor([c.isalnum() or c=='_' for c in vocab]+[False]); seq = [V]+[vocab.index(c) for c in ('ab_2 ab_2+ab_2 A!(2)_ \n'*3+'abcdefghijklmnop')]+[V]; x = torch.tensor([seq])
    rr = R.rows(x,word); letters = [i for i,c in enumerate(vocab) if c.isalpha()]; perm = torch.arange(V+1); perm[letters] = torch.tensor(letters)[torch.randperm(len(letters))]
    hashes = []; words = word.tolist()
    for t,c in enumerate(seq):
        h = hashes[-1] if t and words[c] and words[seq[t-1]] else 0
        hashes.append(((h*1000003+c+1+2**63)%2**64)-2**63)
    for name in ('classes','literal'):
        key,K = keys(vocab,name); ks = key.tolist(); actual = rows3(x,key,K,rr); expected = torch.full_like(actual,-1)
        for t,c in enumerate(seq):
            for j in range(t):
                bits = int(c==seq[j])+2*int(t>=1 and j>=1 and seq[t-1]==seq[j-1])+4*int(hashes[t]==hashes[j])
                expected[0,t,j] = ((ks[c]*K+ks[seq[j+1] if j+1<len(seq) else V])*8+bits)*7+sum(t-j>d for d in (2,4,8,16,32,64))
        assert actual.dtype == torch.int32 and torch.equal(actual,expected), 'head 3 row reference'
        assert torch.equal(actual,rows3(perm[x],key,K,R.rows(perm[x],word))), 'renaming changed head 3'
        assert int(actual.max()) < K*K*8*7 and torch.equal(rr[2][0,0],torch.tensor(seq[1:]+[V])), 'candidate/pad reference'

def run(name, budget=None, smoke=False):
    if budget is not None and budget<128: raise ValueError('budget must be >=128')
    if smoke and (budget is None or budget>256): raise ValueError('smoke requires budget <=256')
    torch.manual_seed(0); vocab = json.loads((I.ROOT/'data/meta.json').read_text())['vocab']; R.reference(vocab); R.R.reference(); reference(vocab)
    torch.cuda.set_per_process_memory_fraction(2.7*2**30/torch.cuda.get_device_properties(0).total_memory); torch.cuda.reset_peak_memory_stats()
    tr,te,V = I.B.load(); cut = int(len(tr)*.95); m = Model(V,name); seconds,chars = R.R.R.fit(m,tr[:cut],budget)
    before = I.stamp(m); renamed,info = I.rename(te); nulls = torch.zeros(4,device='cuda'); old_windows = I.B.N_EVAL_WINDOWS
    try:
        if smoke: I.B.N_EVAL_WINDOWS = 2
        test = I.evaluate(lambda x:m.logprobs(x,nulls),te); ren = I.evaluate(m.logprobs,renamed); val = I.evaluate(m.logprobs,tr[cut:]); R.R.R.causality(m,te); assert I.stamp(m)==before, 'evaluation wrote state'
    finally: I.B.N_EVAL_WINDOWS = old_windows
    settings = dict(seed=0,device='cuda',bs=976,T=128,lr=1e-4,budget=budget,sampling='one_pass',train_end=cut,val_start=cut,requested_chars=cut if budget is None else budget,allocator_limit_gib=2.7,variant=name,clock_variant='clock',inputs=19,first_rows=[n.w.shape[0] for n in m.first],second_inputs=6)
    settings.update(K=m.K,key_map=m.key.tolist(),head_rows=[len(h.W) for h in m.heads],head3_row_dtype='int32',head_lr=.2,head_init=-5.,null_init=0.,adagrad_init=1e-8,batch_delta='sum; per position',probability_floor=1e-6,fallback='ps[4]',sixth_selector='head1_bucket*4+head2_bucket',frozen=False,evaluation_frozen=True,eval_chunk=1024,eval_windows=2 if smoke else old_windows,renaming=info,decision_rules=RULES,smoke=smoke)
    result = dict(test_loss=test,renamed_loss=ren,val_loss=val,renamed_gap=ren-test,train_seconds=seconds,chars=chars,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,head3_rows=len(m.heads[2].W),settings=settings)
    result.update({f'head{i+1}_null_mean':float(nulls[i]/nulls[3]) for i in range(3)})
    with (I.ROOT/f'results/iter13_{"smoke_" if smoke else ""}{name}.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('name',choices=('classes','literal')); p.add_argument('--budget',type=int); p.add_argument('--smoke',action='store_true'); a = p.parse_args(); print(json.dumps(run(a.name,a.budget,a.smoke),indent=2),flush=True)
