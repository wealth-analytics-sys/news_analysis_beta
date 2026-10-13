"""Configuration loading. Env vars for secrets, sources.yaml for behaviour.

Secrets are NEVER committed — see .env.example for the required names.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES_FILE = REPO_ROOT / "sources.yaml"
PROMPTS_DIR = REPO_ROOT / "prompts"
ENV_FILE = REPO_ROOT / ".env"

_ENV_LOADED = False


def install_ssl_truststore() -> str:
    """Make Python trust the OS certificate store.

    Corporate proxies and antivirus HTTPS scanning re-sign traffic with a private root.
    That root lives in the Windows certificate store, which `certifi` (what the SDKs use
    by default) knows nothing about — producing
    'CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain'.

    `truststore` patches Python's SSL to use the OS store instead, which is the correct
    fix — unlike disabling verification, it keeps the connection authenticated.
    Falls back silently if truststore is not installed; set CA_BUNDLE as an alternative.
    """
    import os

    bundle = os.environ.get("CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if bundle:
        os.environ["SSL_CERT_FILE"] = bundle
        os.environ["REQUESTS_CA_BUNDLE"] = bundle
        return f"CA_BUNDLE={bundle}"
    try:
        import truststore

        truststore.inject_into_ssl()
        return "truststore (OS certificate store)"
    except ImportError:
        return "default certifi bundle"


def load_dotenv(path: Path = ENV_FILE, override: bool = False) -> int:
    """Read .env into os.environ. Returns how many keys were loaded.

    Deliberately dependency-free so `pip install python-dotenv` is not another thing that
    can fail. Handles the things people actually do in a .env file:
      - KEY="value"  and  KEY='value'   -> quotes stripped
      - export KEY=value                -> prefix stripped
      - inline  # comments              -> stripped for unquoted values
      - blank lines and full-line comments
    Real environment variables win unless override=True, so CI secrets are never
    shadowed by a stray local file.
    """
    global _ENV_LOADED
    if not path.exists():
        _ENV_LOADED = True
        return 0

    loaded = 0
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not key:
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]                      # quoted: take verbatim
        else:
            val = val.split(" #", 1)[0].strip()  # unquoted: drop trailing comment
        if override or key not in os.environ:
            os.environ[key] = val
            loaded += 1
    _ENV_LOADED = True
    return loaded


def _env(name: str, default: str | None = None, required: bool = False) -> str | None:
    if not _ENV_LOADED:
        load_dotenv()
    val = os.environ.get(name, default)
    if isinstance(val, str):
        val = val.strip().strip('"').strip("'").strip()
    if required and not val:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Set it in .env or in GitHub Actions secrets — never commit it."
        )
    return val or default if val == "" else val


def load_config() -> dict[str, Any]:
    """Build the config dict handed to every stage's run()."""
    if not _ENV_LOADED:
        load_dotenv()
    ssl_mode = install_ssl_truststore()

    with open(SOURCES_FILE, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return {
        "sources": [s for s in raw.get("sources", []) if s.get("enabled", True)],
        "filter": raw.get("filter", {}),
        "cluster": raw.get("cluster", {}),
        # ---- providers -------------------------------------------------
        # Anthropic only (plus `mock` for no-cost tests). Per tier: cheap = S2 triage,
        # S3 clustering, S4 extraction; strong = S6 interpretation (API mode).
        # No embeddings anywhere in the pipeline (BUILD_NOTES D15).
        "llm_provider": _env("LLM_PROVIDER", "anthropic"),
        "llm_provider_cheap": _env("LLM_PROVIDER_CHEAP"),
        "llm_provider_strong": _env("LLM_PROVIDER_STRONG"),
        "anthropic_api_key": _env("ANTHROPIC_API_KEY"),
        "model_cheap_anthropic": _env("MODEL_CHEAP_ANTHROPIC", "claude-haiku-4-5-20251001"),
        "model_strong_anthropic": _env("MODEL_STRONG_ANTHROPIC", "claude-sonnet-5"),
        # ---- S6 HTTPS gateway (BUILD_NOTES D17) --------------------------
        # S6_BACKEND=gateway makes run_s6.py talk to Supabase over HTTPS instead of a direct
        # Postgres connection — used by the Claude cloud routine, which cannot reach 5432.
        "s6_backend": (_env("S6_BACKEND", "db") or "db").lower(),
        "supabase_url": _env("SUPABASE_URL"),
        "supabase_anon_key": _env("SUPABASE_ANON_KEY"),
        "s6_gateway_token": _env("S6_GATEWAY_TOKEN"),
        # ---- database --------------------------------------------------
        # Supabase session-mode pooler (IPv4). Full DSN.
        "database_url": _env("DATABASE_URL"),
        # Portal (S7 UI) auth is managed in users.yaml (name + per-user password), so no
        # portal secret lives in config/.env.
        "dry_run": _env("DRY_RUN", "false").lower() == "true",
        "ssl_mode": ssl_mode,
        # Full-text extraction. Set EXTRACT_FULLTEXT=false to skip article downloads
        # entirely (fast, silent) and run on headline + RSS summary — useful on a
        # locked-down network that blocks article fetches.
        "extract_fulltext": _env("EXTRACT_FULLTEXT", "true").lower() == "true",
        "network_timeout_s": int(_env("NETWORK_TIMEOUT_S", "20")),
        # S1 fetches feeds + extracts article bodies concurrently; this caps the pool.
        "ingest_workers": int(_env("INGEST_WORKERS", "8")),
    }


def load_prompt(name: str) -> str:
    """Load a versioned prompt file. Prompts are never hardcoded in source."""
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")
