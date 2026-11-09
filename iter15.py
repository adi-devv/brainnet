"""Exact layer-local learning, optionally correcting prequential table priors."""
import argparse, json, math, re, time
import torch
import iter14 as N
import iter12 as T
I = N.I
torch.set_grad_enabled(False)
RULES = ['References: backprop 1M 2.330; 6.2M 1.329; 16M 1.124 (renamed 1.530); iter14 dfa 1M 1.992, 6.2M 1.549; iter12 tables 6.2M 1.165 (renamed 1.606, gap 0.441).',
         'local 1M <=1.962 (dfa -0.03): exact local gradient beats random feedback; run local --full.',
         'hybrid beats tables at the same budget by >=0.02 test or >=0.04 renamed: net learns what tables miss; run hybrid --full.',
         'hybrid --full test <1.124: beats backprop at 16M with 2.6x less data, without backprop.',
         'Novel-name learning: (first - repeat) larger for hybrid than tables = better one-shot use of new names.']

def delta(h, e, r): return (e @ r.T) * (1-h.square())

class Model(N.Model):
    def __init__(self, V, hybrid=False):
        super().__init__(V, 3, 1e-3)
        self.feedback = []
        if hybrid:
            for l in range(self.L): self.p[f'R{l}'].zero_(); self.p[f'c{l}'].zero_()
    def train(self, x, y, lp=0):
        if not isinstance(lp,torch.Tensor):
            self.feedback = [self.p[f'R{l}'].T for l in range(self.L)]
            return super().train(x,y)
        hs, zs, us, ids, em = self.forward(x); p = self.p; grads = {}
        final = lp + sum(w*z for w,z in zip(p['g'],zs)); ef = N.error(final,y)
        grads['g'] = torch.stack([(ef*z).sum() for z in zs])
        for l,h in enumerate(hs):
            e = N.error(lp+zs[l],y); d = delta(h,e,p[f'R{l}'])
            grads.update({f'R{l}': N.readout_delta(h,e), f'c{l}': e.sum((0,1)), f'b{l}': d.sum((0,1))})
            if l: grads[f'W{l}'] = N.readout_delta(us[l-1],d)
            else:
                for j in range(6): grads[f'A{j}'] = torch.zeros_like(p[f'A{j}']).index_add_(0,ids[j].flatten(),d.flatten(0,1))
                de = torch.zeros_like(p['E'][x])
                for k in range(3):
                    grads[f'P{k}'] = N.readout_delta(em[:,k],d); de += self.M[k].T @ (d @ p[f'P{k}'].T)
                grads['E'] = torch.zeros_like(p['E']).index_add_(0,x.flatten(),de.flatten(0,1))
        self.step += 1
        for key,v in grads.items():
            assert torch.isfinite(v).all(), key
            v = v/v.norm().clamp_min(1); a,b = self.state[key]
            a.mul_(.9).add_(v,alpha=.1); b.mul_(.999).addcmul_(v,v,value=.001)
            p[key].addcdiv_(a/(1-.9**self.step),(b/(1-.999**self.step)).sqrt().add_(1e-8),value=-self.lr)
        return torch.nn.functional.cross_entropy(final.flatten(0,1),y.flatten(),ignore_index=-1).item()

def plain(o):
    if isinstance(o,torch.Tensor): assert type(o) is torch.Tensor and not o.requires_grad and o.grad_fn is None
    elif isinstance(o,dict):
        for v in o.values(): plain(v)
    elif isinstance(o,(tuple,list)):
        for v in o: plain(v)
    elif hasattr(o,'__dict__'): plain(vars(o))

def checks(net, tables, predict, x):
    assert not torch.is_grad_enabled(); plain(net); plain(tables)
    if net:
        for l in range(net.L):
            a = torch.randn(1,3,256,device='cuda',dtype=torch.float64); y = torch.tensor([[0,-1,net.V-1]],device='cuda')
            r = net.p[f'R{l}'].double(); c = net.p[f'c{l}'].double()
            lp = torch.randn(1,3,net.V,device='cuda',dtype=torch.float64) if tables else 0
            d = delta(a.tanh(),N.error(lp+a.tanh()@r+c,y),r)
            def ce(): return torch.nn.functional.cross_entropy((lp+a.tanh()@r+c).flatten(0,1),y.flatten(),ignore_index=-1)
            for ix in ((0,0,0),(0,1,17),(0,2,255)):
                old = a[ix].item(); eps = 1e-5; a[ix] = old+eps; plus = ce(); a[ix] = old-eps; minus = ce(); a[ix] = old
                torch.testing.assert_close(d[ix],(plus-minus)/(2*eps),atol=1e-8,rtol=1e-5)
    before = I.stamp((net,tables)); other = x.clone(); other[:,47] = (other[:,47]+1) % predict.V
    torch.testing.assert_close(predict(x)[:,:47],predict(other)[:,:47],atol=1e-6,rtol=1e-6)
    assert I.stamp((net,tables)) == before, 'causality evaluation wrote state'

def table_step(m,x,y):
    cod,memory,sel = m.prepare(x)
    for t in range(x.shape[1]):
        keep = y[:,t]>=0; target = y[:,t][keep]
        if not len(target): continue
        c = {n:v[:,t][keep] for n,v in cod.items()}
        m.output(c,[s[:,t][keep] for s in sel],[v[:,t][keep] for v in memory],target); m.base.write(c,target)

def novel(predict, original, renamed, vocab):
    ox,oy = I.B.eval_windows(original); x,y = I.B.eval_windows(renamed); assert ox.shape == x.shape and oy.shape == y.shape
    sums,counts = [0.,0.],[0,0]
    for a in range(0,len(x),16):
        ce = -predict(x[a:a+16].cuda()).log_softmax(-1).gather(-1,y[a:a+16].cuda()[...,None]).squeeze(-1).cpu()
        for b in range(len(ce)):
            text = ''.join(vocab[c] for c in [int(x[a+b,0])]+y[a+b].tolist()); seen = set()
            for match in re.finditer(r'[A-Za-z0-9_]+',text):
                k = int(match.group() in seen)
                for pos in range(max(1,match.start()),match.end()):
                    if y[a+b,pos-1] != oy[a+b,pos-1]: sums[k] += float(ce[b,pos-1]); counts[k] += 1
                seen.add(match.group())
    return dict(novel_first_loss=sums[0]/counts[0] if counts[0] else None,novel_repeat_loss=sums[1]/counts[1] if counts[1] else None,novel_counts=dict(first=counts[0],repeat=counts[1]))

def run(name,budget=999_424,smoke=False):
    if budget is not None and budget<1: raise ValueError('positive budget required')
    if smoke and (budget is None or budget>256): raise ValueError('smoke requires --budget <=256')
    path = I.ROOT/f'results/iter15_{"smoke_"+name if smoke else name+"_"+("full" if budget is None else "1m")}.json'
    if path.exists(): raise FileExistsError(path)
    assert torch.cuda.is_available() and not torch.is_grad_enabled()
    torch.manual_seed(0); torch.cuda.set_per_process_memory_fraction(2.7*2**30/torch.cuda.get_device_properties(0).total_memory); torch.cuda.reset_peak_memory_stats()
    tr,te,V = I.B.load(); cut = int(len(tr)*.95)
    tables = T.Model(V,'learned') if name!='local' else None; net = Model(V,name=='hybrid') if name!='tables' else None
    def prior(x): return torch.cat([tables.logprobs(z) for z in x.split(16)])
    def predict(x,layer=None):
        lp = prior(x) if tables else 0
        if net:
            zs = net.forward(x)[1]; lp = lp + (zs[layer] if layer is not None else sum(w*z for w,z in zip(net.p['g'],zs)))
        return lp.log_softmax(-1)
    predict.V = V; probe = I.B.eval_windows(te)[0][:1].cuda()
    if name=='hybrid': torch.testing.assert_close(predict(probe),prior(probe),atol=1e-6,rtol=1e-6)
    checks(net,tables,predict,probe)
    torch.cuda.synchronize(); start = time.perf_counter(); losses = []; chars = batches = 0
    for x,y in I.batches(tr[:cut],budget,976 if tables else 16,True,V):
        lp = prior(x) if name=='hybrid' else None
        if tables: table_step(tables,x,y)
        if net:
            for a in range(0,len(x),64 if tables else 16):
                n = 64 if tables else 16; losses.append(net.train(x[a:a+n],y[a:a+n],lp[a:a+n] if lp is not None else 0))
        chars += int((y>=0).sum()); batches += 1
    torch.cuda.synchronize(); seconds = time.perf_counter()-start
    if budget is None and net:
        n = max(1,math.ceil(len(losses)*.1)); assert sum(losses[-n:])/n < sum(losses[:n])/n, 'training loss did not decrease'
    checks(net,tables,predict,probe); before = I.stamp((net,tables)); renamed,info = I.rename(te); old = I.B.N_EVAL_WINDOWS
    try:
        if smoke: I.B.N_EVAL_WINDOWS = 2
        test = I.evaluate(predict,te); ren = I.evaluate(predict,renamed); val = I.evaluate(predict,tr[cut:])
        result = novel(predict,te,renamed,json.loads((I.ROOT/'data/meta.json').read_text())['vocab'])
        if net: result['per_layer_test_loss'] = [I.evaluate(lambda x,l=l: predict(x,l),te) for l in range(net.L)]
        assert I.stamp((net,tables)) == before, 'evaluation wrote state'
    finally: I.B.N_EVAL_WINDOWS = old
    settings = dict(seed=0,device='cuda',variant=name,budget=budget,smoke=smoke,bs=976 if tables else 16,net_bs=64 if tables else 16,T=128,D=256,layers=3,
                    lr=1e-3,table_lr=1e-4,optimizer='manual Adam',betas=[.9,.999],eps=1e-8,gradient_clip_norm=1,weight_decay=0,allocator_limit_gib=2.7,
                    sampling='one_pass',train_end=cut,val_start=cut,eval_windows=2 if smoke else old,eval_chunk=16,renaming=info,evaluation_frozen=True,
                    local_rule='own CE with current R.T; no cross-layer derivative',prior='fixed pre-batch tables.logprobs; before all batch writes',
                    readout_init='zero' if name=='hybrid' else 'iter14',g_init=1/3,g_constraint='none',lags=list(range(6)),decays=[.5,.8,.95],embedding_width=32,
                    tables='iter12 learned: bs976; output then base.write per position; lr1e-4; heads Adagrad lr0.2',table_batches=batches if tables else 0,
                    novel_rule='aligned changed targets; maximal ASCII runs over x[0]+y; repeat requires an earlier complete run; empty group=null',decision_rules=RULES)
    result.update(test_loss=test,renamed_loss=ren,renamed_gap=ren-test,val_loss=val,g=net.p['g'].tolist() if net else [],train_seconds=seconds,
                  chars=chars,steps=net.step if net else batches,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,settings=settings)
    with path.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('name',choices=('local','hybrid','tables')); g = p.add_mutually_exclusive_group()
    g.add_argument('--budget',type=int,default=999_424); g.add_argument('--full',action='store_true'); p.add_argument('--smoke',action='store_true'); a = p.parse_args()
    print(json.dumps(run(a.name,None if a.full else a.budget,a.smoke),indent=2),flush=True)
