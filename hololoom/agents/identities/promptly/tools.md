# tools.md — Promptly

## Available
| Tool | Description | Permission |
|------|-------------|------------|
| vault_search | Query PARA vault (read-only) | unrestricted |
| vault_propose | Federation proposal to 00_Inbox/ | unrestricted |
| memory_retrieve | HoloLoom KG query (spring dynamics) | unrestricted |
| memory_store | Write to bus (StoredItem only) | unrestricted |
| file_read | Local filesystem within vault boundaries | unrestricted |
| file_write | Local filesystem within vault boundaries | confirm |
| ollama_generate | Local inference (default model) | unrestricted |

## Forbidden
| Tool | Reason |
|------|--------|
| http_fetch | No external network (soul: privacy > convenience) |
| system_exec | No shell without approval (agent: binary safety) |
| identity_write | Cannot modify agent.md or soul.md directly |

## Usage Patterns (learned)
- vault_search before ollama_generate on factual queries — reduces hallucination
- memory_retrieve with top_k=5 for FAST, top_k=20 for RESEARCH
- file_write requires confirmation for anything outside 01_Projects/

## Performance (updated by heartbeat)
| Tool | Success Rate | Avg Latency | Notes |
|------|-------------|-------------|-------|
| — | — | — | No data yet |
