"""S7 — Serve. Minimalist Streamlit UI.

ZERO business logic here (CLAUDE.md §2 S7). This layer only reads and displays.

Two data modes, chosen automatically:
  DB mode      DATABASE_URL is set -> live pipeline state from Supabase
  Sample mode  no DATABASE_URL     -> renders the Phase 0 sample payload from samples/,
                                      so the card design can be reviewed before S4-S6 exist

Run:  streamlit run src/s7_serve.py
"""
from __future__ import annotations

import html
import json
import os
import socket
import sys
import uuid
from datetime import timedelta, timezone
from pathlib import Path

import streamlit as st

# On Streamlit Community Cloud there is no .env — secrets live in st.secrets. Bridge them
# into os.environ so src.config._env (used by the whole app) finds DATABASE_URL, API keys,
# etc. No-op locally, where .env is used. Never overrides an already-set variable.
try:
    for _k, _v in dict(st.secrets).items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(_k, str(_v))
except Exception:  # noqa: BLE001 — no secrets file locally is fine
    pass


def _esc(s) -> str:
    """Escape untrusted text before it goes into an unsafe_allow_html string.

    Card fields and article titles/urls originate from LLM output stored in the DB;
    escaping keeps a stray '<' or a crafted <a> from breaking the layout or injecting
    markup. Low-risk internal tool, but there is no reason to render it raw.
    """
    return html.escape(str(s if s is not None else ""), quote=True)


def _nd(s) -> str:
    """Escape '$' so Streamlit's markdown does NOT read dollar amounts as LaTeX math.

    Streamlit renders `$...$` as inline LaTeX. Card text is full of dollar figures
    ("$95", "$20.7 billion", "$6.1 billion"), so a pair of them silently turns the text
    between into a spaceless math span — the "20.7billioninforeignexchange" garble. Use for
    PLAIN markdown renders (st.write/st.info/st.warning) that are not already HTML-escaped.
    """
    return str(s if s is not None else "").replace("$", "\\$")


def _txt(s) -> str:
    """HTML-escape AND $-escape — for model text injected inside an unsafe_allow_html
    string. Covers both the layout-break risk (_esc) and the LaTeX-math risk (_nd)."""
    return _esc(s).replace("$", "\\$")


def _fmt_ts(ts) -> str:
    """UTC timestamp -> clean IST display, e.g. '24-Jul-2026 13:07 IST'."""
    if not ts:
        return "—"
    try:
        ist = ts.astimezone(timezone(timedelta(hours=5, minutes=30)))
        return ist.strftime("%d-%b-%Y %H:%M IST")
    except Exception:  # noqa: BLE001
        return str(ts)[:16]


def _fmt_date(ts) -> str:
    """Date only, no time — e.g. '24-June-2026'."""
    if not ts:
        return "—"
    try:
        ist = ts.astimezone(timezone(timedelta(hours=5, minutes=30)))
        return ist.strftime("%d-%B-%Y")
    except Exception:  # noqa: BLE001
        return str(ts)[:10]


# ------------------------------------------------------------------ auth (dummy)
def _load_users() -> list[dict]:
    import yaml
    p = REPO_ROOT / "users.yaml"
    if not p.exists():
        return []
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("users", [])


def _session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex
    return st.session_state["session_id"]


@st.cache_resource
def _login_throttle():
    """ONE throttle for the whole process — shared by every browser session, so opening a
    new session does not reset an account's failed-attempt count (BUILD_NOTES F39)."""
    from src.auth import LoginThrottle
    return LoginThrottle()


def _login_gate(cfg) -> dict:
    """Email + password sign-in against users.yaml (PBKDF2 hashes only — BUILD_NOTES F39).
    Accounts are not listed on the page; 5 failed attempts lock an account for 15 minutes.
    Returns the user dict, or halts the script with st.stop() until signed in."""
    if st.session_state.get("user"):
        return st.session_state["user"]

    from src.auth import verify_password

    st.title("News Interpretation Portal")
    st.caption("Internal research — conditional market analysis, not investment advice. "
               "Sign in to continue.")
    users = _load_users()
    if not users:
        st.error("No users configured. Add entries to users.yaml.")
        st.stop()

    with st.form("login"):
        email = st.text_input("Email").strip()
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        throttle = _login_throttle()
        wait = throttle.locked_for(email)
        if wait:
            st.error(f"Too many failed attempts. Try again in {int(wait // 60) + 1} minute(s).")
            st.stop()
        u = next((x for x in users
                  if str(x.get("email", "")).strip().lower() == email.lower()), None)
        # verify_password runs even for an unknown email, so timing does not reveal accounts.
        if verify_password(pw, (u or {}).get("password_hash")) and u and pw:
            throttle.record_success(email)
            st.session_state["user"] = {"name": u["name"], "email": u.get("email", ""),
                                        "admin": bool(u.get("admin"))}
            _session_id()
            _log_usage(cfg, None, "login")
            st.rerun()
        else:
            throttle.record_failure(email)
            st.error("Incorrect email or password.")
    st.stop()


# ------------------------------------------------------------------ usage logging
def _log_usage(cfg, event_id, action: str) -> None:
    """Write one usage_event, stamped with the signed-in user. Repeatable per-card actions
    (card_opened) are de-duped per session so Streamlit reruns don't inflate counts. Never
    breaks the UI — a logging failure is swallowed."""
    user = st.session_state.get("user") or {}
    email = user.get("email") or "unknown"
    if action == "card_opened":
        seen = st.session_state.setdefault("_usage_seen", set())
        if (action, event_id) in seen:
            return
        seen.add((action, event_id))
    try:
        from src import db
        db.execute(cfg,
                   "INSERT INTO usage_events (user_email, event_id, action, session_id) "
                   "VALUES (%s,%s,%s,%s)",
                   (email, event_id, action, _session_id()))
    except Exception:  # noqa: BLE001 — telemetry must never break the page
        pass


# ------------------------------------------------------------------ category index
_CATEGORY_COLORS = {
    "global_economy": "#378ADD", "global_equity": "#2C6FB0",
    "indian_economy": "#1D9E75", "indian_equity_corporate": "#7F77DD",
    "commodity": "#BA7517", "regulatory": "#993556", "other": "#888780",
}


def _event_category(event_key: str, payload: dict | None) -> str:
    """The card's primary MECE bucket: S6-assigned payload value if present and valid,
    else the deterministic event_key -> category fallback."""
    from src.s2_filter import PRIMARY_CATEGORY_LABELS, primary_category_for_event_key
    cat = ((payload or {}).get("primary_category") or "").strip().lower()
    if cat in PRIMARY_CATEGORY_LABELS:
        return cat
    return primary_category_for_event_key(event_key)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLES = REPO_ROOT / "samples"

st.set_page_config(page_title="News Interpretation Portal", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1150px;}
  h1, h2, h3 {font-weight: 600;}
  .muted {color: #6b7280; font-size: 0.86rem;}
  .pill {display:inline-block; padding:2px 9px; border-radius:11px;
         font-size:0.72rem; border:1px solid; margin-right:5px; white-space:nowrap;}
  .p-neg  {background:#fdecea; color:#b3261e; border-color:#f3c0bb;}
  .p-pos  {background:#e7f5ec; color:#186a3b; border-color:#b7e0c6;}
  .p-warn {background:#fdf4e3; color:#8a6100; border-color:#efd9a8;}
  .p-info {background:#eaf0fb; color:#1c4ea0; border-color:#c2d4f2;}
  .p-grey {background:#f1f3f5; color:#495057; border-color:#dee2e6;}
  .hop {border-left:3px solid #d0d7e2; padding:3px 0 3px 13px; margin:2px 0 2px 4px;}
  .hop-head {font-size:0.88rem; font-weight:600; color:#1f2937;}
  .hop-mech {font-size:0.84rem; color:#4b5563; margin-top:1px; line-height:1.45;}
  .hop-cites {margin-top:4px;}
  .term {background:#f8f9fb; border:1px solid #e5e9f0; border-radius:7px;
         padding:8px 11px; margin-top:7px; font-size:0.85rem;}
  .explain {background:#f4f7fe; border-left:3px solid #6b8fd6; border-radius:5px;
            padding:9px 12px; margin-top:8px; font-size:0.87rem; color:#293241;
            line-height:1.55;}
  .foot {font-size:0.78rem; color:#6b7280; border-top:1px solid #e5e7eb;
         margin-top:14px; padding-top:9px;}
  .card {border:1px solid #e5e7eb; border-radius:10px; padding:16px 18px; margin-bottom:14px;}
  /* ---- event grid tiles (fixed height so all 3 in a row match) ---- */
  .tile {background:#ffffff; border:0.5px solid #e6e8ec; border-left:4px solid #888;
         border-radius:12px; padding:14px 15px 12px; height:132px; box-sizing:border-box;
         display:flex; flex-direction:column; gap:8px; overflow:hidden;
         transition:border-color .12s ease;}
  .tile:hover {border-color:#c3c9d2; border-left-width:4px;}
  .tile-cat {font-size:0.7rem; font-weight:600; letter-spacing:.02em; text-transform:none;}
  .tile-title {font-size:0.95rem; font-weight:600; color:#1f2937; line-height:1.35;
         display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
         overflow:hidden;}
  .tile-meta {font-size:0.73rem; color:#98a0ab; margin-top:auto; display:flex;
         align-items:center; gap:8px; flex-wrap:wrap;}
  /* tighten the gap between a tile and its View button */
  div[data-testid="column"] .stButton {margin-top:-2px;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data access
#
# CACHING (Rushabh, 22-Jul-2026): every st.tabs() body runs on EVERY script rerun —
# Streamlit reruns the whole script top-to-bottom on any widget interaction anywhere on
# the page, not just when the user "changes page". Without caching, clicking a selectbox
# in the Articles tab was also re-querying events, interpretations, per-event articles and
# the ingest log in every OTHER tab, every single time. All DB reads below are wrapped in
# @st.cache_data so a rerun reuses the last result instead of re-hitting Supabase, for
# CACHE_TTL_S seconds or until the user presses "Refresh data" (main(), which calls
# st.cache_data.clear()). Cache key is the DSN string (not the whole `cfg` dict) plus any
# query-specific args — simple, stable, and correctly shared across the whole session.
CACHE_TTL_S = 180   # DB reads are cached this long so most reruns never touch Supabase
                    # (the pooler can be laggy); the Refresh button clears the cache on demand.


def _config():
    try:
        from src.config import load_config
        return load_config()
    except Exception:
        return {}


@st.cache_data(ttl=45, show_spinner=False)
def _db_status(dsn: str):
    """(ok, error_message). Cached only briefly (15s) so a transient blip doesn't lock the
    app into 'no DB' for a full minute — a rerun/Refresh re-tries quickly. Returns the real
    error so the UI can say WHY it can't reach the database instead of guessing."""
    if not dsn:
        return (False, "DATABASE_URL is not set (check your .env).")
    try:
        from src import db
        db.fetch_one({"database_url": dsn}, "SELECT 1 AS ok")
        return (True, "")
    except Exception as exc:  # noqa: BLE001
        return (False, f"{type(exc).__name__}: {str(exc)[:300]}")


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def _cached_article_status_counts(dsn: str) -> list[dict]:
    from src import db
    return db.fetch_all({"database_url": dsn},
                        "SELECT status, COUNT(*) AS n FROM news_articles GROUP BY status")


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def _cached_open_event_count(dsn: str) -> int:
    from src import db
    row = db.fetch_one({"database_url": dsn},
                       "SELECT COUNT(*) AS n FROM news_events WHERE status='open'")
    return (row or {}).get("n", 0)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def _cached_ingest_log(dsn: str) -> list[dict]:
    from src import db
    return db.fetch_all({"database_url": dsn}, """
        SELECT run_at, source, articles_fetched, failures
        FROM ingest_log ORDER BY run_at DESC LIMIT 15
    """)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def _cached_articles(dsn: str, status: str) -> list[dict]:
    from src import db
    where = "" if status == "all" else "WHERE status = %s"
    params = () if status == "all" else (status,)
    return db.fetch_all({"database_url": dsn}, f"""
        SELECT id, source, title, triage_score, status, drop_reason, event_key,
               published_at, (raw_text IS NULL) AS extraction_failed, url
        FROM news_articles {where}
        ORDER BY COALESCE(published_at, ingested_at) DESC LIMIT 200
    """, params)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def _cached_events(dsn: str, only_interpreted: bool = True) -> list[dict]:
    from src import db
    # By default show ONLY events that have an interpretation card — an event with no card
    # is pipeline plumbing, not something an RM should see. Each row carries the latest
    # interpretation's primary_category (for the MECE index) so grouping is one query.
    where = ("WHERE EXISTS (SELECT 1 FROM event_interpretations ei WHERE ei.event_id = e.id)"
             if only_interpreted else "")
    return db.fetch_all({"database_url": dsn}, f"""
        SELECT e.id, e.event_key, e.title, e.last_updated,
               (SELECT payload->>'primary_category' FROM event_interpretations ei
                WHERE ei.event_id = e.id ORDER BY version DESC LIMIT 1) AS primary_category,
               (SELECT payload->'impact_chains'->0->'terminal_impact'->>'direction'
                FROM event_interpretations ei
                WHERE ei.event_id = e.id ORDER BY version DESC LIMIT 1) AS direction
        FROM news_events e
        {where}
        ORDER BY e.last_updated DESC
    """)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def _search_events(dsn: str, term: str) -> list[dict]:
    """Keyword search over interpreted events. Matches the term (case-insensitive) against
    the event title AND the latest interpretation card's text — event_title, summary,
    rm_one_liner and the impact-chain mechanisms/nodes. Returns the SAME row shape as
    _cached_events so the existing tile renderer can be reused unchanged.

    Keyword-only by design (Rushabh, 22-Sep-2026): the pipeline has no embeddings (D15). Only events that HAVE a card are searchable — an event
    with no interpretation is pipeline plumbing, not something an RM searches for."""
    from src import db
    like = f"%{term}%"
    return db.fetch_all({"database_url": dsn}, """
        SELECT e.id, e.event_key, e.title, e.last_updated,
               li.payload->>'primary_category' AS primary_category,
               li.payload->'impact_chains'->0->'terminal_impact'->>'direction' AS direction
        FROM news_events e
        JOIN LATERAL (
            SELECT payload FROM event_interpretations ei
            WHERE ei.event_id = e.id ORDER BY version DESC LIMIT 1
        ) li ON TRUE
        WHERE COALESCE(e.title, '')                       ILIKE %s
           OR COALESCE(li.payload->>'event_title', '')    ILIKE %s
           OR COALESCE(li.payload->>'summary', '')        ILIKE %s
           OR COALESCE(li.payload->>'rm_one_liner', '')   ILIKE %s
           OR (li.payload->'impact_chains')::text         ILIKE %s
        ORDER BY e.last_updated DESC
    """, (like, like, like, like, like))


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def _cached_event_interpretation(dsn: str, event_id: int) -> dict | None:
    from src import db
    return db.fetch_one({"database_url": dsn}, """
        SELECT payload, version, schema_version, citation_warnings, reviewed_by
        FROM event_interpretations WHERE event_id=%s ORDER BY version DESC LIMIT 1
    """, (event_id,))


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def _cached_event_articles(dsn: str, event_id: int) -> list[dict]:
    from src import db
    return db.fetch_all({"database_url": dsn}, """
        SELECT id AS article_id, source, url, title
        FROM news_articles WHERE event_id=%s ORDER BY published_at
    """, (event_id,))


def _db_available(cfg) -> bool:
    ok, _ = _db_status(cfg.get("database_url") or "")
    return ok


@st.cache_data(ttl=60)
def _load_sample(name: str):
    path = SAMPLES / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_sources() -> dict[int, dict]:
    """article_id -> {source, url, title} from the Phase 0 input bundle."""
    data = _load_sample("01_event_input.json") or {}
    return {a["article_id"]: a for a in data.get("articles", [])}


# ------------------------------------------------------------------ renderers
def _dir_pill(direction: str, magnitude: str = "") -> str:
    # magnitude (small/moderate/high) intentionally NOT shown — direction only.
    cls = "p-neg" if direction == "negative" else "p-pos" if direction == "positive" else "p-grey"
    return f'<span class="pill {cls}">{direction}</span>'


def render_card(card: dict, sources: dict[int, dict], show_title: bool = True,
                rm_view: bool = False):
    # rm_view hides the technical/provenance bits from relationship managers (fact ids,
    # source-agreement, KB data-gaps, unverified tags). Admins see the full card.
    if show_title:
        st.markdown(f"### {_nd(card.get('event_title','(untitled event)'))}")

    meta = [
        f'<span class="pill p-info">{_txt(card.get("event_type","-"))}</span>',
    ]
    if card.get("is_update"):
        meta.append('<span class="pill p-warn">updated</span>')
    st.markdown(" ".join(meta), unsafe_allow_html=True)

    st.write("")
    st.markdown(_nd(card.get("summary", "")))

    if card.get("rm_one_liner"):
        st.info(f"**RM one-liner** — {_nd(card['rm_one_liner'])}")

    # ---- impact chains
    st.markdown("#### Impact chains")
    for i, chain in enumerate(card.get("impact_chains", []), 1):
        with st.container(border=True):
            st.markdown(f'**Impact {i}**')
            for hop in chain.get("chain", []):
                cites = ""
                if not rm_view:  # fact ids / evidence chips are hidden from RMs
                    if hop.get("fact_ids"):
                        cites += " ".join(
                            f'<span class="pill p-info">fact {fid}</span>' for fid in hop["fact_ids"]
                        )
                    for ev in hop.get("evidence", []) or []:
                        cites += (f'<span class="pill p-warn">{_txt(ev.get("metric"))} '
                                  f'{_txt(ev.get("value"))} (art {_txt(ev.get("article_id"))})</span>')
                cites_div = f'<div class="hop-cites">{cites}</div>' if cites else ""
                st.markdown(
                    f'<div class="hop">'
                    f'<div class="hop-head">{_txt(hop.get("from"))} &rarr; {_txt(hop.get("to"))}</div>'
                    f'<div class="hop-mech">{_txt(hop.get("mechanism",""))}</div>'
                    f'{cites_div}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            t = chain.get("terminal_impact", {}) or {}
            st.markdown(
                f'<div class="term"><b>{_txt(t.get("asset","-"))}</b> &nbsp;'
                f'{_dir_pill(t.get("direction","-"), t.get("magnitude","-"))}<br>'
                f'<span class="muted">Conditional on: {_txt(t.get("conditionality","-"))}</span></div>',
                unsafe_allow_html=True,
            )

            # "What this means" — the plain-language explanation under the chain (v1.3).
            if chain.get("plain_explanation"):
                st.markdown(
                    f'<div class="explain"><b>What this means:</b> '
                    f'{_txt(chain["plain_explanation"])}</div>',
                    unsafe_allow_html=True,
                )

    # ---- source agreement (hidden from RMs)
    sa = card.get("source_agreement", {}) or {}
    consistent, discrepancies = sa.get("consistent_facts", []), sa.get("discrepancies", [])
    if not rm_view and (consistent or discrepancies):
        st.markdown("#### Source agreement")
        for c in consistent:
            arts = ", ".join(str(a) for a in c.get("article_ids", []))
            st.markdown(f"- ✔ {_txt(c.get('claim'))} &nbsp;<span class='muted'>(articles {arts})</span>",
                        unsafe_allow_html=True)
        for d in discrepancies:
            vals = " · ".join(f"art {v.get('article_id')}: {_nd(v.get('value'))}"
                              for v in d.get("values", []))
            st.warning(f"**Discrepancy** — {_nd(d.get('claim'))} → {vals}")

    # ---- historical patterns (was never rendered before v1.4)
    patterns = card.get("historical_patterns_cited", []) or []
    if patterns:
        st.markdown("#### Historical context")
        for hp in patterns:
            attr = hp.get("attribution", "")
            vs = hp.get("verification_status", "")
            # RMs see just the pattern; the attribution ("... (unverified)") and the
            # 'unverified' tag are internal signals, hidden from them.
            if rm_view:
                extra = ""
            else:
                tag = (" <span class='pill p-grey'>unverified</span>"
                       if vs == "model_knowledge" else "")
                extra = f" <span class='muted'>— {_txt(attr)}</span>{tag}"
            st.markdown(f"- {_txt(hp.get('pattern',''))}{extra}", unsafe_allow_html=True)

    # ---- conditional actions
    actions = card.get("investor_actions_conditional", [])
    if actions:
        st.markdown("#### Conditional observations")
        for a in actions:
            st.markdown(
                f"- **If** {_txt(a.get('if'))} → {_txt(a.get('then'))}",
                unsafe_allow_html=True,
            )

    if card.get("relevant_fund_categories"):
        st.markdown("**Relevant fund categories**")
        st.markdown(" ".join(
            f'<span class="pill p-grey">{_txt(c)}</span>' for c in card["relevant_fund_categories"]
        ), unsafe_allow_html=True)
    # Data gaps for KB is an internal curation queue — hidden from RMs.
    if not rm_view and card.get("data_gaps_for_kb"):
        with st.expander(f"Data gaps for KB ({len(card['data_gaps_for_kb'])})"):
            for g in card["data_gaps_for_kb"]:
                st.markdown(f"- {_nd(g)}")

    # ---- footnotes then sources (URLs resolved by join, never stored in payload)
    for marker, meaning in (card.get("footnotes") or {}).items():
        st.markdown(f'<div class="foot">{_txt(marker)} {_txt(meaning)}</div>',
                    unsafe_allow_html=True)

    if sources:
        links = " &nbsp;·&nbsp; ".join(
            f'<a href="{_esc(a["url"])}" target="_blank" rel="noopener noreferrer">'
            f'{_esc(a["source"])} — {_esc(a["title"])}</a>'
            for a in sources.values()
        )
        st.markdown(f'<div class="foot"><b>Sources:</b> {links}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ views
def view_pipeline_health(cfg, live: bool):
    st.subheader("Pipeline health")
    if not live:
        st.info("No database configured — showing sample data only. "
                "Set DATABASE_URL to see live pipeline state.")
        return

    dsn = cfg.get("database_url") or ""
    counts = _cached_article_status_counts(dsn)
    by = {c["status"]: c["n"] for c in counts}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("New", by.get("new", 0))
    c2.metric("Kept", by.get("kept", 0))
    c3.metric("Dropped", by.get("dropped", 0))
    c4.metric("Open events", _cached_open_event_count(dsn))

    st.markdown("**Last ingest runs**")
    runs = _cached_ingest_log(dsn)
    if runs:
        st.dataframe(runs, use_container_width=True, hide_index=True)
    else:
        st.caption("No ingest runs recorded yet.")


def view_articles(cfg, live: bool):
    st.subheader("Article inspector")
    st.caption("Use this to sanity-check the triage threshold before trusting it.")
    if not live:
        st.info("Requires a database connection.")
        return

    status = st.selectbox("Status", ["kept", "dropped", "new", "all"], index=0)
    rows = _cached_articles(cfg.get("database_url") or "", status)
    st.caption(f"{len(rows)} article(s)")
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True,
                     column_config={"url": st.column_config.LinkColumn("url", width="small")})


def view_events(cfg, live: bool):
    st.subheader("Event feed")

    if not live:
        st.error("Can't reach the database right now, so no events can be shown.")
        # Raw driver errors can name the pooler host and project user — admins only (F43).
        if (st.session_state.get("user") or {}).get("admin"):
            ok, err = _db_status(cfg.get("database_url") or "")
            st.caption(f"Reason: {err}" if err else "Unknown database error.")
            st.caption("Check DATABASE_URL and your connection (the Supabase session pooler "
                       "can briefly refuse connections), then press Refresh. Interpretation "
                       "cards already exist in the database — this is a connectivity issue, "
                       "not missing data.")
        else:
            st.caption("This is usually a brief connection issue. Please press Refresh in a "
                       "minute.")
        with st.expander("Preview the card layout (sample, not live data)"):
            card = _load_sample("04_interpret_output.json")
            if card:
                # rm_view=True: the full view opens a 'Data gaps' expander, and Streamlit
                # forbids nested expanders — this crashed the outage page (F42).
                render_card(card, _sample_sources(), rm_view=True)
        return

    from src.s2_filter import PRIMARY_CATEGORIES, PRIMARY_CATEGORY_LABELS

    dsn = cfg.get("database_url") or ""
    # NB: ?event=<id> detail routing is handled once in main(), before the tabs — not here.

    events = _cached_events(dsn, only_interpreted=True)
    if not events:
        st.caption("No interpreted events yet.")
        return

    for ev in events:
        ev["_cat"] = _event_category(ev.get("event_key", ""),
                                     {"primary_category": ev.get("primary_category")})
    present = [c for c in PRIMARY_CATEGORIES if any(e["_cat"] == c for e in events)]

    # ---- category dropdown ----
    options = ["All"] + present
    counts = {c: sum(1 for e in events if e["_cat"] == c) for c in present}
    def _opt_label(c):
        return f"All categories ({len(events)})" if c == "All" \
            else f"{PRIMARY_CATEGORY_LABELS[c]} ({counts[c]})"
    sel = st.selectbox("Category", options, format_func=_opt_label,
                       label_visibility="collapsed")
    st.write("")

    filtered = events if sel == "All" else [e for e in events if e["_cat"] == sel]

    _render_grid(cfg, filtered, ctx=f"feed_{sel}")


TILE_PAGE = 60   # tiles per page; "Show more" reveals the next page


def _render_grid(cfg, events: list[dict], ctx: str) -> None:
    """Responsive 3-column tile grid (Streamlit stacks columns to 1 on mobile), paginated so
    the page stays fast as cards accumulate. `ctx` namespaces widget keys: Streamlit runs
    every tab body on each rerun, so the Events and Search tabs can render the same event."""
    shown_key = f"shown_{ctx}"
    shown = st.session_state.get(shown_key, TILE_PAGE)
    cols = st.columns(3, gap="medium")
    for i, ev in enumerate(events[:shown]):
        with cols[i % 3]:
            _render_tile(cfg, ev, ctx)
    if len(events) > shown:
        if st.button(f"Show more ({len(events) - shown} more)", key=f"more_{ctx}",
                     use_container_width=True):
            st.session_state[shown_key] = shown + TILE_PAGE
            st.rerun()


def _render_tile(cfg, ev, ctx: str = "feed") -> None:
    """A minimal, color-coded event tile + a View button that routes to the detail page."""
    from src.s2_filter import PRIMARY_CATEGORY_LABELS
    cat = ev.get("_cat", "other")
    color = _CATEGORY_COLORS.get(cat, "#888780")
    direction = (ev.get("direction") or "").strip().lower()
    dchip = ""
    if direction in ("negative", "positive", "mixed", "unclear"):
        cls = ("p-neg" if direction == "negative" else
               "p-pos" if direction == "positive" else "p-grey")
        dchip = f'<span class="pill {cls}">{direction}</span>'
    st.markdown(
        f'<div class="tile" style="border-left-color:{color}">'
        f'<div class="tile-cat" style="color:{color}">{_txt(PRIMARY_CATEGORY_LABELS.get(cat, cat))}</div>'
        f'<div class="tile-title">{_txt(ev.get("title") or ev["event_key"])}</div>'
        f'<div class="tile-meta">Updated at {_fmt_date(ev.get("last_updated"))} {dchip}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.button("View details →", key=f"view_{ctx}_{ev['id']}", use_container_width=True):
        _log_usage(cfg, ev["id"], "card_opened")
        st.query_params["event"] = str(ev["id"])
        st.rerun()


def _render_detail(cfg, dsn, ev_id) -> None:
    """Single-card detail view reached via ?event=<id>, with a Back button. Renders only
    this one card (fast) and mounts the read-to-bottom sentinel."""
    try:
        eid = int(ev_id)
    except (TypeError, ValueError):
        st.query_params.clear()
        st.rerun()
        return

    if st.button("← Back to events", key="detail_back"):
        if "event" in st.query_params:
            del st.query_params["event"]
        st.rerun()

    interp = _cached_event_interpretation(dsn, eid)
    if not interp:
        st.warning("This event has no interpretation yet.")
        return
    arts = _cached_event_articles(dsn, eid)
    src_map = {a["article_id"]: a for a in arts}
    is_admin = bool((st.session_state.get("user") or {}).get("admin"))
    render_card(interp["payload"], src_map, show_title=True, rm_view=not is_admin)


def view_search(cfg, live: bool):
    """Keyword search over interpreted events + their cards. Reuses the event feed's tile
    grid and the ?event=<id> detail routing, so a search hit opens the same card view."""
    st.subheader("Search news")

    if not live:
        st.error("Can't reach the database right now, so search is unavailable.")
        if (st.session_state.get("user") or {}).get("admin"):
            st.caption("Check DATABASE_URL and connectivity, then press Refresh.")
        else:
            st.caption("This is usually a brief connection issue. Please press Refresh in a "
                       "minute.")
        return

    dsn = cfg.get("database_url") or ""
    # NB: ?event=<id> detail routing is handled once in main(), before the tabs — not here.

    # The term is kept in a plain session key (not the widget's own key): the widget is not
    # rendered while a card is open, and Streamlit drops widget state it doesn't render, so
    # "Back" would otherwise come back to an empty search.
    term = st.text_input(
        "Search", value=st.session_state.get("_search_term", ""),
        placeholder="Search events by keyword — e.g. crude, RBI, FPI, rupee, IT sector",
        label_visibility="collapsed").strip()
    if term != st.session_state.get("_search_term"):
        st.session_state["_search_term"] = term
        st.session_state.pop("shown_search", None)   # new term starts at page 1
    if not term:
        st.caption("Type a keyword to search event titles, summaries, RM one-liners and "
                   "impact chains across all interpreted events.")
        return
    if len(term) < 2:
        st.caption("Enter at least 2 characters.")
        return

    from src.s2_filter import PRIMARY_CATEGORIES  # noqa: F401 (kept for parity/ordering)
    results = _search_events(dsn, term)
    if not results:
        st.info(f"No interpreted events match “{term}”.")
        return

    st.caption(f"{len(results)} event(s) matching “{term}”")
    for ev in results:
        ev["_cat"] = _event_category(ev.get("event_key", ""),
                                     {"primary_category": ev.get("primary_category")})
    _render_grid(cfg, results, ctx="search")


# ------------------------------------------------------------------ diagnostics
#
# NOTE ON CLAUDE.md §2 ("S7 holds zero business logic"): this tab is a deliberate,
# operator-only exception, added at Rushabh's request to make the funnel observable when
# "nothing comes through". It does NOT reimplement any stage — it only calls the existing
# src/ stage functions (s1_ingest fetch primitives, s2_filter._pass_a/_pass_b) and displays
# their results. No persistence happens in the in-memory funnel; the full run reuses each
# stage's own run().
def _fetch_entries(sources: list[dict], limit_per_feed: int):
    """In-memory RSS fetch — no DB. Reuses S1's feed backend and UA. Returns
    (articles, per_source_report, backend_name)."""
    from src.s1_ingest import FEED_BACKEND, USER_AGENT, _parse_published, feedparser

    socket.setdefaulttimeout(20)
    articles: list[dict] = []
    per_source: list[dict] = []
    aid = 1
    for src in sources:
        name, url = src.get("name"), src.get("url")
        before = len(articles)
        err = None
        try:
            feed = feedparser.parse(url, agent=USER_AGENT)
            for e in list(getattr(feed, "entries", []) or [])[:limit_per_feed]:
                link = (e.get("link") or "").strip()
                title = (e.get("title") or "").strip()
                if not link or not title:
                    continue
                articles.append({
                    "id": aid, "source": name, "url": link, "title": title,
                    "rss_summary": ((e.get("summary") or e.get("description") or "")[:400]
                                    or None),
                    "raw_text": None,
                    "published_at": _parse_published(e),
                })
                aid += 1
        except Exception as exc:  # noqa: BLE001 — one bad feed must not stop the others
            err = f"{type(exc).__name__}: {exc}"[:160]
        per_source.append({"source": name, "fetched": len(articles) - before, "error": err})
    return articles, per_source, FEED_BACKEND


def view_diagnostics(cfg, live: bool):
    st.subheader("Pipeline diagnostics — see the funnel")
    st.caption(
        "Runs S1 fetch → S2 triage **in memory (no database writes)** so you can see exactly "
        "what arrived, what was dropped and why, and what survived. This is the fastest way "
        "to find where 'nothing comes through'."
    )

    provider = cfg.get("llm_provider_cheap") or cfg.get("llm_provider") or "?"
    threshold = float(cfg.get("filter", {}).get("triage_threshold", 7))
    st.caption(
        f"Triage provider (cheap tier): **{provider}** &nbsp;·&nbsp; keep-threshold: "
        f"**{threshold:g}** &nbsp;·&nbsp; an article is KEPT if max(portfolio, kb) ≥ threshold."
    )

    sources = cfg.get("sources", [])
    if not sources:
        st.error("No sources configured — check sources.yaml.")
        return

    names = [s.get("name") for s in sources]
    chosen = st.multiselect("Feeds to test", names, default=names)
    limit = st.slider("Articles per feed", 1, 20, 8)

    if not st.button("▶ Run funnel  (fetch → triage, no DB)", type="primary"):
        st.info("Pick feeds and press the button. Nothing is written to the database.")
        return

    chosen_sources = [s for s in sources if s.get("name") in chosen]

    # ---- S1 fetch ---------------------------------------------------------
    with st.status("S1 — fetching feeds…", expanded=True) as status:
        try:
            articles, per_source, backend = _fetch_entries(chosen_sources, limit)
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            return
        st.write(f"feed backend: `{backend}`")
        st.dataframe(per_source, use_container_width=True, hide_index=True)
        status.update(label=f"S1 — fetched {len(articles)} article(s)", state="complete")

    if not articles:
        st.error(
            "**Nothing was fetched.** Every selected feed returned zero usable articles — "
            "this is a network / feed problem (blocked RSS, proxy, or a changed feed URL), "
            "**not** a triage decision. Check the per-source errors above. If you are on a "
            "locked-down office network, try from a normal connection, or set "
            "`EXTRACT_FULLTEXT=false` and retry."
        )
        return

    from src import s2_filter
    fcfg = cfg.get("filter", {})

    # ---- S2 pass A (deterministic) ---------------------------------------
    survivors, dropped_a = s2_filter._pass_a(articles, fcfg)
    drop_a_map = {aid: reason for aid, reason in dropped_a}

    # ---- S2 pass B (LLM triage) ------------------------------------------
    with st.status("S2 — triage scoring…", expanded=False) as status:
        try:
            scores, held = s2_filter._pass_b(cfg, survivors)
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            st.error("Triage raised before returning. Usually a bad/missing key for the "
                     f"cheap-tier provider `{provider}`. Set LLM_PROVIDER_CHEAP=mock to "
                     "prove the rest of the plumbing offline.")
            return
        status.update(label=f"S2 — scored {len(scores)}/{len(survivors)}", state="complete")

    held_set = set(held)
    kept_ids = [a["id"] for a in survivors
                if (s := scores.get(a["id"])) and s["score"] >= threshold]

    # ---- funnel metrics ---------------------------------------------------
    m = st.columns(4)
    m[0].metric("Fetched", len(articles))
    m[1].metric("Survived pass A", len(survivors),
                delta=(-(len(articles) - len(survivors)) or None))
    m[2].metric("Scored", len(scores), delta=(f"{len(held)} held" if held else None),
                delta_color="inverse")
    m[3].metric("KEPT", len(kept_ids))

    # ---- per-article verdict table ---------------------------------------
    rows = []
    for a in articles:
        aid = a["id"]
        pr = kb = sc = ek = None
        if aid in drop_a_map:
            verdict, reason = "❌ dropped · A", drop_a_map[aid]
        elif aid in held_set:
            verdict, reason = "⏳ held", "triage infra error — not a verdict"
        elif aid in scores:
            s = scores[aid]
            pr, kb, sc, ek = (s["portfolio_relevance"], s["kb_value"], s["score"],
                              s["event_key"])
            if sc >= threshold:
                verdict, reason = "✅ kept", s.get("event_type", "")
            else:
                verdict, reason = "❌ dropped · B", f"max score {sc:g} < {threshold:g}"
        else:
            verdict, reason = "❓ unscored", "model returned no score for this id"
        rows.append({
            "id": aid, "source": a["source"], "title": a["title"][:72],
            "verdict": verdict, "portfolio": pr, "kb": kb, "score": sc,
            "event_key": ek, "reason": reason,
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # ---- one-line diagnosis ----------------------------------------------
    if held:
        st.error(
            f"**{len(held)} article(s) were HELD, not scored.** The triage LLM call failed "
            f"(TLS, rate-limit, or a bad/missing key for provider `{provider}`). This is a "
            f"classic 'nothing comes through' cause: an infrastructure error is being "
            f"surfaced, not a relevance decision. Fix the key/provider, or set "
            f"`LLM_PROVIDER_CHEAP=mock` in .env to confirm the plumbing works offline."
        )
    elif not kept_ids:
        st.warning(
            "**Everything was dropped at triage — but nothing errored.** Every article "
            "scored below the threshold. Two possibilities: (1) a genuinely low-signal news "
            "batch, or (2) the triage prompt is under-scoring durable/structural stories "
            "(BUILD_NOTES **F17**). Read the portfolio/kb columns above — if structural "
            "stories (LNG contracts, import-duty changes) are sitting at 5–6, that's F17, "
            "not a real 'no news' day."
        )
    else:
        st.success(f"**{len(kept_ids)} article(s) would flow into clustering.** The funnel "
                   f"is healthy through triage. Event keys that would form:")
        keys: dict[str, list[int]] = {}
        for a in survivors:
            s = scores.get(a["id"])
            if s and s["score"] >= threshold:
                keys.setdefault(s["event_key"], []).append(a["id"])
        for k, v in sorted(keys.items(), key=lambda x: -len(x[1])):
            st.markdown(f"- `{k}` → {len(v)} article(s): {v}")
        if live:
            st.caption("These would become events on a full `--write` run. Use the Events "
                       "tab to see interpretation cards after the pipeline persists them.")
        else:
            st.caption("No database is connected, so this stops at triage. Connect "
                       "DATABASE_URL to run clustering → extraction → interpretation and "
                       "produce cards.")


# ------------------------------------------------------------------ store & interpret
def view_feedback(cfg, live: bool):
    st.subheader("Share feedback")
    st.caption("Tell us what's useful and what's missing — it shapes what we build next.")
    if not live:
        st.info("Feedback needs a database connection.")
        return

    dsn = cfg.get("database_url") or ""
    events = _cached_events(dsn, only_interpreted=True)
    about_options = {"General feedback (not about a specific event)": None}
    for ev in events[:60]:
        about_options[(ev.get("title") or ev["event_key"])[:80]] = ev["id"]

    with st.form("feedback_form", clear_on_submit=True):
        rating = st.radio("How useful is this to you?",
                          ["Very useful", "Useful", "Neutral", "Not useful"],
                          index=1, horizontal=True)
        about = st.selectbox("About", list(about_options.keys()))
        comment = st.text_area("Comments (optional)",
                               placeholder="What would make this more useful for your client work?")
        submitted = st.form_submit_button("Send feedback")

    if submitted:
        user = st.session_state.get("user") or {}
        try:
            from src import db
            db.execute(cfg,
                       "INSERT INTO feedback (user_email, event_id, rating, comment) "
                       "VALUES (%s,%s,%s,%s)",
                       (user.get("email", "unknown"), about_options.get(about),
                        rating, (comment or None)))
            st.success("Thanks — your feedback was recorded.")
        except Exception:  # noqa: BLE001
            st.error("Couldn't save that right now. Try again in a moment.")


# ------------------------------------------------------------------ main
def main():
    cfg = _config()
    user = _login_gate(cfg)          # halts the script until signed in
    live = _db_available(cfg)

    title_col, user_col = st.columns([6, 2])
    with title_col:
        st.title("News Interpretation Portal")
    with user_col:
        st.write("")
        st.caption(f"Signed in as {user['name']}")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Refresh", use_container_width=True,
                         help=f"Reuses cached data for up to {CACHE_TTL_S}s. Press to pull "
                              f"the latest events/cards immediately."):
                st.cache_data.clear()
                st.rerun()
        with b2:
            if st.button("Sign out", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    st.markdown(
        "<span class='muted'>Internal research — conditional market analysis, not investment "
        "advice.</span>", unsafe_allow_html=True)
    st.write("")

    # ---- detail routing (top level, exactly ONCE) ----
    # A ?event=<id> query param opens the single-card detail view. It is handled HERE, before
    # the tabs, rather than inside view_events/view_search — Streamlit executes every tab body
    # on each rerun, so routing it inside both tabs rendered the card (and its 'detail_back'
    # button, key="detail_back") twice, raising StreamlitDuplicateElementKey.
    ev_id = st.query_params.get("event")
    if ev_id:
        if not live:
            st.error("Can't reach the database right now, so this event can't be shown. "
                     "Please press Refresh in a minute.")
        else:
            _render_detail(cfg, cfg.get("database_url") or "", ev_id)
        return

    if user.get("admin"):
        tabs = st.tabs(["Events", "Search", "Feedback", "🔬 Diagnostics",
                        "Pipeline health", "Articles"])
        with tabs[0]:
            view_events(cfg, live)
        with tabs[1]:
            view_search(cfg, live)
        with tabs[2]:
            view_feedback(cfg, live)
        with tabs[3]:
            view_diagnostics(cfg, live)
        with tabs[4]:
            view_pipeline_health(cfg, live)
        with tabs[5]:
            view_articles(cfg, live)
    else:
        tabs = st.tabs(["Events", "Search", "Feedback"])
        with tabs[0]:
            view_events(cfg, live)
        with tabs[1]:
            view_search(cfg, live)
        with tabs[2]:
            view_feedback(cfg, live)


main()
