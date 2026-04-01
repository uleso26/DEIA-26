# Security Notes

## Local configuration

- Keep `.env` local. The repository only ships `.env.example`.
- Live ingestion makes outbound requests to public data sources when the matching `USE_LIVE_*_INGESTION` flags are enabled.
- Local LLM and embedding support uses Ollama on your machine when it is reachable. If Ollama is unavailable, the platform falls back to deterministic routing, synthesis, or lexical retrieval.

## Sensitive data

- Do not commit API keys, tokens, or local credentials into notebooks, source files, or saved verification artifacts.
- Verification artifacts and lineage files should be shared only after checking that they do not contain local absolute paths or secrets.

## Runtime posture

- The project is designed as a coursework MVP, not a hardened production deployment.
- SQLite, file-backed fallbacks, and local logs are used for reproducibility and should be replaced with managed controls in a production environment.
