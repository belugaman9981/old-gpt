"""Tiny character-level GPT trained ONLY on the .txt files in ./data
Usage:
  python old_gpt.py            # train, then print a sample
  python old_gpt.py gen "Whan that Aprille"   # sample from saved model
"""
import glob, re, sys, time, torch, torch.nn as nn, torch.nn.functional as F

# ---- config (sized for an 8-core CPU, ~3M params) ----
BLOCK, N_EMBD, N_HEAD, N_LAYER = 128, 192, 6, 6
BATCH, LR, ITERS, EVAL_EVERY = 32, 6e-4, 5000, 250
torch.manual_seed(1)
torch.set_num_threads(8)

def load_text():
    text = ""
    for f in sorted(glob.glob("data/*.txt")):
        t = open(f, encoding="utf-8", errors="ignore").read()
        m = re.search(r"\*\*\* ?START OF[^\n]*\*\*\*", t)   # strip Gutenberg header
        if m: t = t[m.end():]
        e = re.search(r"\*\*\* ?END OF", t)                  # ...and footer
        if e: t = t[:e.start()]
        text += t.strip() + "\n\n"
        print(f"loaded {f}: {len(t):,} chars")
    return text

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.ff = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
                                nn.Linear(4 * N_EMBD, N_EMBD))
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = [z.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2) for z in (q, k, v)]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # attention
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.ff(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, N_EMBD)
        self.pos = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab)
    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        logits = self.head(self.ln(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n, temp=0.8):
        for _ in range(n):
            logits, _ = self(idx[:, -BLOCK:])
            probs = F.softmax(logits[:, -1] / temp, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

def sample(model, stoi, itos, prompt="\n", n=500):
    prompt = "".join(c for c in prompt if c in stoi) or "\n"
    idx = torch.tensor([[stoi[c] for c in prompt]])
    return "".join(itos[i] for i in model.generate(idx, n)[0].tolist())

if len(sys.argv) > 1 and sys.argv[1] == "gen":
    ck = torch.load("old_gpt.pt")
    stoi, itos = ck["stoi"], ck["itos"]
    model = GPT(len(stoi)); model.load_state_dict(ck["model"]); model.eval()
    print(sample(model, stoi, itos, sys.argv[2] if len(sys.argv) > 2 else "\n", 800))
    sys.exit()

text = load_text()
assert len(text) > 100_000, "Put more .txt files in ./data (aim for 1 MB+)"
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text])
cut = int(0.9 * len(data))
train, val = data[:cut], data[cut:]
print(f"{len(text):,} chars, vocab {len(chars)}")

def batch(d):
    ix = torch.randint(len(d) - BLOCK - 1, (BATCH,))
    return (torch.stack([d[i:i + BLOCK] for i in ix]),
            torch.stack([d[i + 1:i + BLOCK + 1] for i in ix]))

model = GPT(len(chars))
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
opt = torch.optim.AdamW(model.parameters(), lr=LR)
t0 = time.time()
for it in range(ITERS + 1):
    if it % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            vl = sum(model(*batch(val))[1].item() for _ in range(10)) / 10
        model.train()
        print(f"iter {it}: val loss {vl:.3f}  ({time.time() - t0:.0f}s)")
        torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos}, "old_gpt.pt")
    _, loss = model(*batch(train))
    opt.zero_grad(); loss.backward(); opt.step()

model.eval()
print("\n--- sample ---\n" + sample(model, stoi, itos))