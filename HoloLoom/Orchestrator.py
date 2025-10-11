from __future__ import annotations
import os, json, math, warnings, hashlib, asyncio, time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Callable
from pathlib import Path
from collections import deque
import numpy as np
import networkx as nx
from typing import Protocol

# Optional deps (same as original)
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None
    warnings.warn("spaCy model not found; falling back to regex motifs.")

try:
    from sentence_transformers import SentenceTransformer
    _SENT = SentenceTransformer(os.environ.get("HOLOLOOM_BASE_ENCODER", "all-MiniLM-L6-v2"))
except Exception:
    _SENT = None
    warnings.warn("sentence-transformers not available; using random embeddings.")

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None
    warnings.warn("rank-bm25 not available; BM25 disabled.")

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh
    _HAVE_SCIPY = True
except Exception:
    eigsh = None
    _HAVE_SCIPY = False
    warnings.warn("SciPy not available; using dense numpy eigensolver.")

import torch
import torch.nn as nn
import torch.nn.functional as F

# Data Contracts (expanded from original)
@dataclass
class Config:
    scales: List[int] = field(default_factory=lambda: [96, 192, 384])
    fusion_weights: Dict[int, float] = field(default_factory=lambda: {96: 0.25, 192: 0.35, 384: 0.40})
    memory_path: Optional[str] = None
    base_model_name: Optional[str] = None
    fast_mode: bool = False
    mode: str = 'Fused'  # 'Bare', 'Fast', 'Fused'

@dataclass
class MemoryShard:
    id: str
    text: str
    episode: str
    entities: List[str] = field(default_factory=list)
    motifs: List[str] = field(default_factory=list)
    scales: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KGEdge:
    src: str
    dst: str
    type: str
    weight: float = 1.0
    span_id: Optional[str] = None

@dataclass
class Query:
    text: str
    h: int = field(init=False)

    def __post_init__(self):
        self.h = hash(self.text)

@dataclass
class Features:
    psi: np.ndarray
    motifs: List[str]
    metrics: Dict[str, float]
    confidence: float = 1.0

    @classmethod
    def empty(cls):
        return cls(psi=np.zeros(6), motifs=[], metrics={}, confidence=0.0)

@dataclass
class Context:
    hits: List[Tuple[MemoryShard, float]]
    kg_sub: nx.MultiDiGraph
    shard_texts: List[str]
    relevance: float = 0.0

    @classmethod
    def empty(cls):
        return cls(hits=[], kg_sub=nx.MultiDiGraph(), shard_texts=[], relevance=0.0)

@dataclass
class ActionPlan:
    chosen_tool: str
    adapter: str
    tool_probs: Dict[str, float]

@dataclass
class Response:
    query: str
    motifs: List[str]
    entities: List[str]
    hits: List[Tuple[str, str]]
    psi: List[float]
    psi_metrics: Dict[str, float]
    tool_probs: Dict[str, float]
    chosen_tool: str
    adapter: str
    metadata: Dict = field(default_factory=dict)

# Component Interfaces
class Embedder(Protocol):
    async def features(self, kg_sub: nx.MultiDiGraph, shard_texts: List[str], emb: 'MatryoshkaEmbeddings') -> Tuple[np.ndarray, Dict[str, float]]:
        ...

class MotifDetector(Protocol):
    async def extract(self, text: str) -> List[str]:
        ...

class Retriever(Protocol):
    async def search(self, query: str, k: int = 6, fast: bool = False) -> List[Tuple[MemoryShard, float]]:
        ...

class PolicyEngine(Protocol):
    async def decide(self, features: Features, context: Context) -> ActionPlan:
        ...

class KGStore(Protocol):
    def add_edge(self, e: KGEdge): ...
    def subgraph_for_entities(self, ents: List[str]) -> nx.MultiDiGraph: ...

# Existing classes adapted
class MotifExtractor:
    RULES = [
        (r"because|since|so that", "cause→effect"),
        (r"but|however|yet", "contrast"),
        (r"if .* then", "condition→consequence"),
        (r"goal|aim|so we can", "goal→constraint"),
        (r"why|how", "question→answer"),
    ]

    def __init__(self, force_regex: bool = False):
        self.force_regex = force_regex

    async def extract(self, text: str) -> List[str]:
        motifs: List[str] = []
        if not self.force_regex and _NLP:
            doc = _NLP(text)
            for sent in doc.sents:
                if any(t.dep_ == "mark" for t in sent):
                    motifs.append("subordinate→main")
                if any(t.dep_ == "advcl" for t in sent):
                    motifs.append("advcl→main")
        import re
        lowered = text.lower()
        for pat, name in self.RULES:
            if re.search(pat, lowered):
                motifs.append(name)
        return sorted(set(motifs))

class MatryoshkaEmbeddings:
    def __init__(self, sizes: List[int] = [96, 192, 384], base_model_name: Optional[str] = None,
                 external_heads: Optional[Dict[int, Callable[[np.ndarray], np.ndarray]]] = None):
        assert sorted(sizes) == sizes, "sizes must be ascending"
        self.sizes = sizes
        self.external_heads = external_heads or {}
        if _SENT and base_model_name:
            try:
                globals()['_SENT'] = SentenceTransformer(base_model_name)
            except Exception as e:
                warnings.warn(f"Failed to load {base_model_name}, falling back: {e}")
        if _SENT:
            v = _SENT.encode(["probe"], normalize_embeddings=True)[0]
            self.base_dim = int(len(v))
        else:
            self.base_dim = sizes[-1]
        self._build_projection(seed=12345)
        self._last_hash = None

    def _build_projection(self, seed: int = 12345):
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.normal(size=(self.base_dim, self.base_dim)))
        self.proj = {d: Q[:, :d] for d in self.sizes}

    def refresh_runtime_qr(self, corpus_texts: List[str]):
        # use SHA-256 for stability
        h = hashlib.sha256("\n".join(corpus_texts).encode("utf-8")).hexdigest()
        if h != self._last_hash:
            seed = int(h[:8], 16)
            self._build_projection(seed=seed)
            self._last_hash = h

    def encode_base(self, texts: List[str]) -> np.ndarray:
        if _SENT:
            return _SENT.encode(texts, normalize_embeddings=True)
        vecs = []
        for t in texts:
            seed = abs(hash(t)) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.normal(0, 1, self.base_dim)
            v = v / (np.linalg.norm(v) + 1e-9)
            vecs.append(v)
        return np.vstack(vecs)

    def encode_scales(self, texts: List[str], size: Optional[int] = None) -> Dict[int, np.ndarray] | np.ndarray:
        if len(texts) == 0:
            # Guard empty input
            if size is not None:
                return np.zeros((0, size))
            return {d: np.zeros((0, d)) for d in self.sizes}
        base = self.encode_base(texts)
        if size is not None:
            if size in self.external_heads:
                return self.external_heads[size](base)
            return base @ self.proj[size]
        out: Dict[int, np.ndarray] = {}
        for d in self.sizes:
            out[d] = self.external_heads[d](base) if d in self.external_heads else (base @ self.proj[d])
        return out

class KG:
    def __init__(self):
        self.G = nx.MultiDiGraph()

    def add_edge(self, e: KGEdge):
        self.G.add_node(e.src)
        self.G.add_node(e.dst)
        self.G.add_edge(e.src, e.dst, type=e.type, weight=e.weight, span_id=e.span_id)

    def subgraph_for_entities(self, ents: List[str]) -> nx.MultiDiGraph:
        nodes = set()
        for ent in ents:
            if ent in self.G:
                nodes.add(ent)
                nodes.update(self.G.successors(ent))
                nodes.update(self.G.predecessors(ent))
        return self.G.subgraph(nodes).copy()

class RetrieverMS:
    def __init__(self, shards: List[MemoryShard], emb: MatryoshkaEmbeddings, fusion_weights: Optional[Dict[int, float]] = None):
        self.shards = shards
        self.emb = emb
        self.texts = [s.text for s in shards]
        self.emb.refresh_runtime_qr(self.texts)
        self.vecs_per_scale: Dict[int, np.ndarray] = {d: emb.encode_scales(self.texts, size=d) for d in emb.sizes}
        self.fusion_weights = fusion_weights or {d: 1.0 / len(emb.sizes) for d in emb.sizes}
        if BM25Okapi:
            tokenized = [t.lower().split() for t in self.texts]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def _norm(self, s: np.ndarray) -> np.ndarray:
        mu, sd = float(s.mean()), float(s.std() + 1e-9)
        z = (s - mu) / sd
        return 1.0 / (1.0 + np.exp(-z))

    async def search(self, query: str, k: int = 6, fast: bool = False) -> List[Tuple[MemoryShard, float]]:
        if fast:
            d = min(self.emb.sizes)
            mat = self.vecs_per_scale[d]
            q = self.emb.encode_scales([query], size=d)[0]
            s = mat @ q
            s = self._norm(s)
            idx = np.argsort(-s)[:k]
            return [(self.shards[i], float(s[i])) for i in idx]
        fused = np.zeros(len(self.texts))
        for d, mat in self.vecs_per_scale.items():
            q = self.emb.encode_scales([query], size=d)[0]
            s = mat @ q
            fused += self.fusion_weights.get(d, 0.0) * self._norm(s)
        if self.bm25:
            bm = self.bm25.get_scores(query.lower().split())
            fused = 0.85 * fused + 0.15 * self._norm(bm)
        idx = np.argsort(-fused)[:k]
        return [(self.shards[i], float(fused[i])) for i in idx]

class SpectralFusion:
    async def features(self, kg_sub: nx.MultiDiGraph, shard_texts: List[str], emb: MatryoshkaEmbeddings) -> Tuple[np.ndarray, Dict[str, float]]:
        n = kg_sub.number_of_nodes()
        if n > 1:
            nodes = list(kg_sub.nodes())
            idx = {u: i for i, u in enumerate(nodes)}
            rows, cols, data = [], [], []
            for u, v, d in kg_sub.edges(data=True):
                w = float(d.get("weight", 1.0))
                rows.extend([idx[u], idx[v]])
                cols.extend([idx[v], idx[u]])
                data.extend([w, w])
            if _HAVE_SCIPY:
                A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
                D = np.asarray(A.sum(axis=1)).ravel()
                L = np.diag(D) - A.toarray()
                k = min(4, n - 1)
                try:
                    vals, _ = eigsh(L.astype(float), k=k, which='SM')
                    vals = np.sort(vals)
                except Exception:
                    vals = np.linalg.eigvalsh(L)
                    vals = np.sort(vals)
            else:
                A = np.zeros((n, n))
                for r, c, w in zip(rows, cols, data):
                    A[r, c] += w
                D = np.diag(A.sum(axis=1))
                L = D - A
                vals = np.linalg.eigvalsh(L)
                vals = np.sort(vals)
            spec = np.pad(vals[:4], (0, max(0, 4 - len(vals))), constant_values=0.0)
            fiedler = float(spec[1]) if len(spec) > 1 else 0.0
        else:
            spec = np.zeros(4)
            fiedler = 0.0
        if not shard_texts:
            psi = np.concatenate([spec, np.zeros(2)])
            metrics = {"fiedler": fiedler, "topic_var": 0.0, "coherence": 1.0 if fiedler > 1e-6 else 0.0}
            return psi, metrics
        V = emb.encode_scales(shard_texts, size=max(emb.sizes))
        try:
            m = min(len(V), 64)
            if m == 0:
                topic = np.zeros(2); tv = 0.0
            else:
                _, s, _ = np.linalg.svd(V[:max(16, m), :], full_matrices=False)
                topic = s[:2]
                tv = float((topic.sum() / (s.sum() + 1e-9)))
        except Exception:
            topic = np.zeros(2)
            tv = 0.0
        psi = np.concatenate([spec, topic])
        metrics = {"fiedler": fiedler, "topic_var": tv, "coherence": (1.0 if fiedler > 1e-6 else 0.0) + tv}
        return psi, metrics

# Neural Core
def maybe_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomMHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x, gates: torch.Tensor):
        B, T, D = x.shape
        H = self.n_heads
        Dh = self.d_head
        q = self.Wq(x).view(B, T, H, Dh)
        k = self.Wk(x).view(B, T, H, Dh)
        v = self.Wv(x).view(B, T, H, Dh)
        attn = torch.einsum('bthd,bshd->bhts', q, k) / math.sqrt(Dh)
        A = torch.softmax(attn, dim=-1)
        g = gates.view(B, H, 1, 1)
        A = A * g
        z = torch.einsum('bhts,bshd->bthd', A, v).contiguous().view(B, T, D)
        return self.Wo(z), A

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x, mem):
        assert mem.size(-1) == self.d_model, f"CrossAttention mem dim {mem.size(-1)} != d_model {self.d_model}"
        B, T, D = x.shape
        M = mem.size(1)
        H = self.n_heads
        Dh = self.d_head
        q = self.Wq(x).view(B, T, H, Dh)
        k = self.Wk(mem).view(B, M, H, Dh)
        v = self.Wv(mem).view(B, M, H, Dh)
        attn = torch.einsum('bthd,bmhd->bhtm', q, k) / math.sqrt(Dh)
        A = torch.softmax(attn, dim=-1)
        z = torch.einsum('bhtm,bmhd->bthd', A, v).contiguous().view(B, T, D)
        return self.Wo(z)

class MotifGatedMHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, n_motifs: int = 8):
        super().__init__()
        self.mha = CustomMHA(d_model, n_heads)
        self.gate_proj = nn.Linear(n_motifs, n_heads)

    def forward(self, x, motif_ctrl):
        gates = torch.sigmoid(self.gate_proj(motif_ctrl))
        out, attn = self.mha(x, gates)
        return out, attn

class LoRALikeFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 1024, r: int = 8, n_adapters: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.adapters = nn.ModuleList([nn.Sequential(nn.Linear(d_model, r, bias=False), nn.Linear(r, d_model, bias=False)) for _ in range(n_adapters)])

    def forward(self, x, adapter_idx: int = 0):
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        h = h + self.adapters[adapter_idx](x)
        return h

class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model=384, n_heads=4, n_motifs=8, n_adapters=4):
        super().__init__()
        self.cross = CrossAttention(d_model, n_heads)
        self.mha = MotifGatedMHA(d_model, n_heads, n_motifs)
        self.ffn = LoRALikeFFN(d_model, d_ff=4 * d_model, r=16, n_adapters=n_adapters)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, mem, motif_ctrl, adapter_idx):
        x = x + self.cross(self.ln1(x), mem)
        mha_out, _ = self.mha(self.ln2(x), motif_ctrl)
        x = x + mha_out
        x = x + self.ffn(self.ln3(x), adapter_idx)
        return x

class NeuralCore(nn.Module):
    def __init__(self, d_model=384, n_layers=2, n_heads=4, n_motifs=8, n_adapters=4):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(1, 16, d_model) / math.sqrt(d_model))
        self.blocks = nn.ModuleList([TinyTransformerBlock(d_model, n_heads, n_motifs, n_adapters) for _ in range(n_layers)])
        self.readout = nn.Linear(d_model, d_model)
        self.tool_head = nn.Linear(d_model, 4)
        self.tools = ["answer", "search", "notion_write", "calc"]

    async def decide(self, mem: torch.Tensor, ctrl: torch.Tensor, adapter_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B = mem.size(0)
        x = self.latent.expand(B, -1, -1)
        for blk in self.blocks:
            x = blk(x, mem, ctrl, adapter_idx)
        pooled = x.mean(dim=1)
        logits = self.tool_head(self.readout(pooled))
        return logits, pooled

class TSBandit:
    def __init__(self, n_arms: int):
        self.success = np.ones(n_arms)
        self.fail = np.ones(n_arms)

    def choose(self) -> int:
        samples = np.random.beta(self.success, self.fail)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        if reward > 0:
            self.success[arm] += reward
        else:
            self.fail[arm] += -reward

class PDVClient:
    def __init__(self, root: str = "data"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    async def store_shard(self, shard: MemoryShard):
        with (self.root / "pdv_shards.jsonl").open('a', encoding='utf-8') as f:
            f.write(json.dumps(shard.__dict__) + "\n")

    async def derive_matryoshka(self, text: str, emb: MatryoshkaEmbeddings) -> Dict[str, List[float]]:
        vecs = emb.encode_scales([text])
        return {str(k): v[0].tolist() for k, v in vecs.items()}

class MemoAIClient:
    def __init__(self, root: str = "data"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.root / "memoai_vectors.jsonl"

    async def upsert_vectors(self, shard_id: str, scale_vectors: Dict[str, List[float]]):
        rec = {"id": shard_id, "vectors": scale_vectors, "ts": int(time.time())}
        with self.vec_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + "\n")

# MemoryManager
class MemoryManager:
    def __init__(self, shards: List[MemoryShard], emb: MatryoshkaEmbeddings, cfg: Config):
        self.working_memory = {}
        self.episodic_buffer = deque(maxlen=100)
        self.retriever = RetrieverMS(shards, emb, cfg.fusion_weights)
        self.kg = KG()
        self.pdv = PDVClient()
        self.memo = MemoAIClient()
        self.persistence_queue = asyncio.Queue()
        self._run_archiver()

    def _run_archiver(self):
        async def archiver():
            while True:
                item = await self.persistence_queue.get()
                q_shard = item['q_shard']
                await asyncio.gather(
                    self.pdv.store_shard(q_shard),
                    self.memo.upsert_vectors(q_shard.id, q_shard.scales)
                )
                self.persistence_queue.task_done()

        asyncio.create_task(archiver())

    async def retrieve(self, query: Query, fast: bool) -> Context:
        q_hash = query.h
        if q_hash in self.working_memory:
            return self.working_memory[q_hash]

        hits = await self.retriever.search(query.text, k=6, fast=fast)
        shard_texts = [s.text for s, _ in hits]
        entities = [w for w in query.text.split() if w[0].isupper()]
        for ent in entities[:3]:
            self.kg.add_edge(KGEdge(src=ent, dst="query", type="MENTIONS", weight=1.0))
        kg_sub = self.kg.subgraph_for_entities(entities[:3])

        context = Context(hits=hits, kg_sub=kg_sub, shard_texts=shard_texts, relevance=0.5)
        self.working_memory[q_hash] = context
        return context

    async def persist(self, query: Query, results: Dict, features: Features):
        self.episodic_buffer.append({
            'query': query.text,
            'results': results,
            'features': features,
            'timestamp': time.time()
        })

        motifs = features.motifs
        entities = [w for w in query.text.split() if w[0].isupper()]
        q_shard = MemoryShard(id=f"q_{query.h}", text=query.text, episode="query", entities=entities, motifs=motifs,
                              scales={str(d): self.retriever.emb.encode_scales([query.text], size=d)[0].tolist() for d in self.retriever.emb.sizes})
        await self.persistence_queue.put({'q_shard': q_shard})

# ToolExecutor
class ToolExecutor:
    def __init__(self, bandit: TSBandit):
        self.bandit = bandit
        self.tools = ["answer", "search", "notion_write", "calc"]

    async def execute(self, plan: ActionPlan, query: Query) -> Dict:
        return {"chosen_tool": plan.chosen_tool, "probs": plan.tool_probs}

# UnifiedPolicy
class UnifiedPolicy:
    def __init__(self, core: NeuralCore, psi_proj: nn.Linear, device: torch.device, adapter_for_dim: Dict[int, int], adapter_bank: Dict[int, str], mem_dim: int, emb: MatryoshkaEmbeddings):
        self.core = core
        self.psi_proj = psi_proj
        self.device = device
        self.adapter_for_dim = adapter_for_dim
        self.adapter_bank = adapter_bank
        self.mem_dim = mem_dim
        self.emb = emb
        self.bandit = TSBandit(n_arms=4)

    async def decide(self, features: Features, context: Context) -> ActionPlan:
        if not context.shard_texts:
            mem = torch.zeros(1, 1, self.mem_dim).to(self.device)
        else:
            mem_np = self.emb.encode_scales(context.shard_texts, size=self.mem_dim)
            mem = torch.tensor(mem_np).unsqueeze(0).to(self.device)

        psi = torch.tensor(features.psi, dtype=torch.float32).unsqueeze(0).to(self.device)
        motif_ctrl = self._ctrl_from(features.motifs).to(self.device)
        ctrl = torch.clamp(motif_ctrl + 0.25 * torch.sigmoid(self.psi_proj(psi)), 0, 1)

        episodes = len(set(s.episode for s, _ in context.hits)) if context.hits else 0
        reward = features.metrics.get("coherence", 0.0) + 0.1 * episodes
        arm = self.bandit.choose()

        logits, _ = await self.core.decide(mem, ctrl, self.adapter_for_dim.get(self.mem_dim, 0))
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        tool_idx = int(np.argmax(probs))
        tool = self.core.tools[tool_idx]
        adapter_idx = self.adapter_for_dim.get(self.mem_dim, 0)
        adapter = self.adapter_bank.get(adapter_idx, "general")

        self.bandit.update(arm, reward)

        return ActionPlan(chosen_tool=tool, adapter=adapter, tool_probs={self.core.tools[i]: float(probs[i]) for i in range(len(probs))})

    def _ctrl_from(self, motifs: List[str]) -> torch.Tensor:
        motif_vocab = [
            "cause→effect", "contrast", "condition→consequence", "goal→constraint",
            "question→answer", "subordinate→main", "advcl→main", "setup→twist"
        ]
        vec = np.zeros(len(motif_vocab), dtype=np.float32)
        for m in motifs:
            if m in motif_vocab:
                vec[motif_vocab.index(m)] = 1.0
        return torch.tensor(vec).unsqueeze(0)

# Orchestrator
class HoloLoomOrchestrator:
    def __init__(self, shards: List[MemoryShard], cfg: Config):
        self.emb = MatryoshkaEmbeddings(sizes=cfg.scales, base_model_name=cfg.base_model_name)
        force_regex = (cfg.mode == 'Bare')
        self.motif_detector = MotifExtractor(force_regex=force_regex)
        self.spectral = SpectralFusion()
        self.memory = MemoryManager(shards, self.emb, cfg)
        self.device = maybe_device()
        self.mem_dim = max(cfg.scales) if not cfg.fast_mode else min(cfg.scales)
        self.core = NeuralCore(d_model=self.mem_dim).to(self.device)
        self.psi_proj = nn.Linear(6, 8).to(self.device)
        self.policy = UnifiedPolicy(
            core=self.core,
            psi_proj=self.psi_proj,
            device=self.device,
            adapter_for_dim={min(cfg.scales): 1, sorted(cfg.scales)[1]: 2, max(cfg.scales): 3},
            adapter_bank={0: "general", 1: "farm", 2: "brewing", 3: "mirrorcore"},
            mem_dim=self.mem_dim,
            emb=self.emb
        )
        self.tool_engine = ToolExecutor(TSBandit(n_arms=4))
        self.mode = cfg.mode
        self.fast_retrieval = (self.mode != 'Fused')

    async def process(self, query: Query) -> Response:
        try:
            async with asyncio.timeout(5.0):
                return await self._optimal_path(query)
        except asyncio.TimeoutError:
            return await self._fast_path(query)
        except Exception as e:
            return self._safe_fallback(query, e)

    async def _optimal_path(self, query: Query) -> Response:
        motif_task = asyncio.create_task(self.motif_detector.extract(query.text))
        context = await self.memory.retrieve(query, fast=self.fast_retrieval)
        motifs = await motif_task

        if self.mode == 'Bare':
            psi = np.zeros(6)
            metrics = {'coherence': 0.0, 'fiedler': 0.0, 'topic_var': 0.0}
        else:
            psi, metrics = await self.spectral.features(context.kg_sub, context.shard_texts, self.emb)

        features = Features(psi=psi, motifs=motifs, metrics=metrics)

        plan = await self.policy.decide(features, context)

        if plan.chosen_tool == 'answer':
            results = {'direct': True, 'chosen_tool': plan.chosen_tool, 'probs': plan.tool_probs}
        else:
            results = await self.tool_engine.execute(plan, query)

        response = self._generate_response(query, features, context, plan, results)
        asyncio.create_task(self.memory.persist(query, results, features))
        return response

    async def _fast_path(self, query: Query) -> Response:
        motifs = await self.motif_detector.extract(query.text)
        features = Features.empty()
        features.motifs = motifs
        context = Context.empty()
        plan = ActionPlan(chosen_tool="answer", adapter="general", tool_probs={"answer": 1.0, "search": 0.0, "notion_write": 0.0, "calc": 0.0})
        results = await self.tool_engine.execute(plan, query)
        return self._generate_response(query, features, context, plan, results)

    def _safe_fallback(self, query: Query, error: Exception) -> Response:
        print(f"Error: {error}")
        return Response(
            query=query.text,
            motifs=[],
            entities=[],
            hits=[],
            psi=[],
            psi_metrics={},
            tool_probs={},
            chosen_tool="error",
            adapter="none",
            metadata={"error": str(error)}
        )

    def _generate_response(self, query: Query, features: Features, context: Context, plan: ActionPlan, results: Dict) -> Response:
        hits_summary = [(h[0].id, h[0].text[:60] + ("…" if len(h[0].text) > 60 else "")) for h in context.hits]
        entities = [w for w in query.text.split() if w[0].isupper()]
        return Response(
            query=query.text,
            motifs=features.motifs,
            entities=entities,
            hits=hits_summary,
            psi=features.psi.tolist(),
            psi_metrics=features.metrics,
            tool_probs=plan.tool_probs,
            chosen_tool=plan.chosen_tool,
            adapter=plan.adapter
        )

# Persistence helpers
async def persist_memory_v3(shards: List[MemoryShard], path: str, schema: str = "hololoom_poc_v3", encoder: Optional[str] = None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        for s in shards:
            rec = {
                "schema": schema,
                "created": int(time.time()),
                "encoder": encoder or os.environ.get("HOLOLOOM_BASE_ENCODER", "all-MiniLM-L6-v2"),
                "id": s.id,
                "text": s.text,
                "episode": s.episode,
                "entities": s.entities,
                "motifs": s.motifs,
                "scales": s.scales,
            }
            f.write(json.dumps(rec) + "\n")

def load_memory_v3(path: str) -> List[MemoryShard]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[MemoryShard] = []
    for line in p.read_text(encoding='utf-8').splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        shard_fields = {
            "id": obj.get("id", ""),
            "text": obj.get("text", ""),
            "episode": obj.get("episode", ""),
            "entities": obj.get("entities") or [],
            "motifs": obj.get("motifs") or [],
            "scales": obj.get("scales") or {},
        }
        out.append(MemoryShard(**shard_fields))
    return out

# Demo
DEMO_SHARDS = [
    MemoryShard(id="s1", text="Explain multi‑head self‑attention using a joke with double meaning.", episode="MC_2025-10-10", entities=["self-attention"], motifs=["setup→twist"]),
    MemoryShard(id="s2", text="ColBERT uses late interaction where query and doc tokens interact via MaxSim.", episode="MC_2025-10-10", entities=["ColBERT"], motifs=["contrast"]),
    MemoryShard(id="s3", text="Farm low‑ABV compute: pin hot shards in RAM and use PQ/IVF compression.", episode="FarmOps", entities=["Farm"], motifs=["goal→constraint"]),
    MemoryShard(id="s4", text="Brewing SNA schedule: small nutrient doses at 24/48 hours to reduce sulfur.", episode="BrewLab", entities=["Brewing"], motifs=["cause→effect"]),
    MemoryShard(id="s5", text="Phase space U(ψ) detects cycles; use recurrence plots to catch stuck loops.", episode="MirrorCore", entities=["U(ψ)"], motifs=["question→answer"]),
    MemoryShard(id="s6", text="Matryoshka embeddings enable nested coarse→fine representations per span.", episode="MirrorCore", entities=["Matryoshka"], motifs=["condition→consequence"]),
]

async def main():
    cfg = Config(
        scales=[96, 192, 384],
        fusion_weights={96: 0.25, 192: 0.35, 384: 0.40},
        memory_path="data/memory_shards.jsonl",
        base_model_name=os.environ.get("HOLOLOOM_BASE_ENCODER"),
        fast_mode=False,
        mode='Fused'
    )
    loaded = load_memory_v3(cfg.memory_path) if cfg.memory_path else []
    shards = loaded or DEMO_SHARDS
    orchestrator = HoloLoomOrchestrator(shards, cfg)
    queries = [
        "How does multi‑head attention resolve a pun compared to ColBERT?",
        "Design a frost plan for fig trees and low‑ABV compute scheduling.",
        "Why did my cider get sulfur and how does SNA help?",
    ]
    for q_text in queries:
        query = Query(text=q_text)
        response = await orchestrator.process(query)
        print("\n== QUERY ==\n", q_text)
        print(json.dumps(response.__dict__, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())
