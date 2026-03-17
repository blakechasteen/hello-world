# Nervous System Architecture (Cross-Reference)

HoloLoom's role in the distributed agent ecosystem is documented canonically in
the Femtoclaw repository:

**Canonical document**: `nanoclaw/docs/NERVOUS_SYSTEM.md`
(WSL path: `/home/blake/nanoclaw/docs/NERVOUS_SYSTEM.md`)

## HoloLoom's Role: The Cortex

HoloLoom is the cortex of the nervous system — deep reasoning, multi-pass
refinement, Thompson Sampling model selection. It engages when the reflex arc
(Ollama direct) is insufficient.

### HoloLoom's Responsibilities in the Ecosystem

1. **Deliberate thought**: Multi-pass refinement with convergence detection
2. **Model routing**: Thompson Sampling bandit selecting across local and rig models
3. **Episodic memory (Bus)**: Tier 2 memory — `StoredItem` protocol over the memory bus
4. **Vault bridge**: Read-only search + federation proposals to PARA vault
5. **TTS synthesis**: Chatterbox client for voice output
6. **Jenny visualization**: Adaptive viz runtime for Matrix messages

### Protocols HoloLoom Participates In

| Protocol | HoloLoom's Role |
|----------|----------------|
| **Runner** | Implements `hololoom` backend — called by Femtoclaw's runner registry |
| **EpisodicStore** | Implements over memory bus — both runtimes read/write |
| **VaultFederation** | Proposes notes via `vault_bridge.py` |

### Key Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /promptly/chat` | Primary chat (soul, 20-turn memory, multi-pass) |
| `POST /elle/chat` | Elle chat (single-pass, calm operational tone) |
| `GET /memory/store` | Episodic store write (planned) |
| `GET /memory/query` | Episodic store query (planned) |
| `POST /promptly/tts` | Text-to-speech via Chatterbox |
| `GET /promptly/vault/search` | Vault keyword search |
| `POST /vault/propose` | Vault federation proposal |

### Degradation Behavior

When HoloLoom is down, Femtoclaw falls back to the reflex arc (Ollama direct).
Multi-pass refinement, Thompson Sampling routing, and episodic memory writes are
lost. The system still talks.

See the full degradation matrix in the canonical document.

---

*This is a pointer, not the source of truth. For the complete architecture,*
*read `nanoclaw/docs/NERVOUS_SYSTEM.md`.*
