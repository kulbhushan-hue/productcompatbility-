"""
z/OS Product Compatibility Bot — single-file Streamlit app.

Loads `zos_compat_bot.pkl` (built by build_pickle.py from data/*.csv, which
is scraped from IBM's and Broadcom's PUBLIC compatibility pages — no login
required, see README.md) and answers: "which products are compatible with
z/OS version X?" using an exact structured lookup + a Claude-written summary
grounded in that lookup (RAG).

Run locally:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    streamlit run app.py

Deploy: push this repo to GitHub, then on share.streamlit.io point a new
app at app.py and add ANTHROPIC_API_KEY under Secrets.
"""

import os
import re

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

import anthropic

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="z/OS Compatibility Bot", page_icon="🖥️", layout="wide")

CLAUDE_MODEL = "claude-sonnet-4-6"
ZOS_VERSION_PATTERN = re.compile(r"z/?os\s*v?(\d+)r(\d+)", re.IGNORECASE)

SYSTEM_PROMPT = """You are a mainframe (z/OS) product-compatibility assistant.
You are given:
1. A STRUCTURED, exact list of products confirmed compatible with a given z/OS
   version (this is ground truth -- trust it completely for the YES/NO facts).
2. Some RETRIEVED supporting notes/context (PTF requirements, caveats, source
   links) that may add useful nuance.

Write a clear, well-organized answer for the user:
- Group results by vendor (IBM, Broadcom).
- List each compatible product with its version and any caveat from the notes.
- If the structured list is empty, say so plainly and suggest checking the
  product name spelling or a different z/OS version -- do not invent products.
- Always mention that this reflects the last-scraped data and the user should
  verify against the live vendor page for anything going into production,
  and include the source URLs you were given.
- Never state a product is compatible unless it appears in the structured list.
- Keep it concise: a short intro line, then a clean grouped list.
"""


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def normalize_zos_version(raw: str) -> str:
    raw = raw.strip()
    m = ZOS_VERSION_PATTERN.search(raw.replace(".", "R").replace("-", "R"))
    if m:
        return f"Z/OS V{m.group(1)}R{m.group(2)}"
    m2 = re.search(r"(\d)[.\sRr]?(\d)", raw)
    if m2:
        return f"Z/OS V{m2.group(1)}R{m2.group(2)}"
    return raw.upper()


@st.cache_resource
def load_artifact(path: str = "zos_compat_bot.pkl"):
    return joblib.load(path)


def products_compatible_with(df, zos_version, product_filter=None, vendor_filter=None):
    target = normalize_zos_version(zos_version)
    mask = (df["item_norm"] == target) & (df["status"].str.upper() == "YES")
    if vendor_filter and vendor_filter != "Any":
        mask &= df["vendor"].str.lower() == vendor_filter.lower()
    if product_filter:
        mask &= df["product_name"].str.contains(product_filter, case=False, na=False)
    cols = ["vendor", "product_name", "product_version", "category", "item", "status", "notes", "source_url"]
    return df.loc[mask, cols].drop_duplicates().sort_values(["vendor", "product_name"])


def retrieve(query, vectorizer, matrix, chunk_texts, chunk_meta, top_k=8):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return [
        {"text": chunk_texts[i], **chunk_meta[i]}
        for i in top_idx
        if sims[i] > 0
    ]


def generate_answer(zos_version, structured_matches, retrieved_chunks, product_filter=None, api_key=None):
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    if structured_matches.empty:
        structured_text = "No compatible products found in the structured database for this z/OS version" + (
            f" and product filter '{product_filter}'." if product_filter else "."
        )
    else:
        structured_text = structured_matches.to_string(index=False)

    context_text = "\n".join(
        f"- [{c['vendor']} | {c['product_name']}] {c['text']} (source: {c['source_url']})"
        for c in retrieved_chunks
    )

    user_prompt = f"""z/OS version requested: {zos_version}
Product filter (if any): {product_filter or "none - list all compatible products"}

STRUCTURED MATCHES (ground truth):
{structured_text}

RETRIEVED SUPPORTING CONTEXT:
{context_text if context_text else "(none retrieved)"}

Write the final answer for the user now."""

    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in resp.content if hasattr(block, "text"))


def safe_generate_answer(zos_version, structured_matches, retrieved_chunks, product_filter=None, api_key=None):
    """Wraps generate_answer so a bad/missing key shows a friendly message
    instead of crashing the whole app with a raw traceback."""
    try:
        return generate_answer(zos_version, structured_matches, retrieved_chunks, product_filter, api_key), None
    except anthropic.AuthenticationError:
        return None, (
            "❌ **Anthropic API key is invalid.** Double-check it in Streamlit "
            "Secrets (or the box in the sidebar) — it should start with "
            "`sk-ant-` and have no extra spaces or quotes. "
            "Generate/verify keys at console.anthropic.com → API Keys."
        )
    except anthropic.PermissionDeniedError:
        return None, "❌ This key doesn't have permission to use this model. Check your Anthropic account's plan/model access."
    except anthropic.RateLimitError:
        return None, "⏳ Rate limit or quota hit on this API key. Check usage/billing at console.anthropic.com."
    except anthropic.BadRequestError as e:
        if "credit balance" in str(e).lower():
            return None, (
                "💳 **Anthropic account credit balance is too low.** "
                "Go to console.anthropic.com → Plans & Billing → add a payment "
                "method or purchase credits, then try again. (Note: a claude.ai "
                "subscription is separate from API billing.)"
            )
        return None, f"❌ Bad request to Claude API: {e}"
    except anthropic.APIConnectionError:
        return None, "🌐 Couldn't reach the Anthropic API — check network/firewall settings on the host."
    except Exception as e:
        return None, f"❌ Unexpected error calling Claude: {e}"


# ----------------------------------------------------------------------
# Load artifact
# ----------------------------------------------------------------------
artifact = load_artifact()
df = artifact["df"]
vectorizer = artifact["vectorizer"]
matrix = artifact["matrix"]
chunk_texts = artifact["chunk_texts"]
chunk_meta = artifact["chunk_meta"]

known_zos_versions = sorted(
    df.loc[df["item"].str.contains("z/os", case=False, na=False), "item_norm"].dropna().unique().tolist()
)
known_products = sorted(df["product_name"].dropna().unique().tolist())

# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🖥️ z/OS Product Compatibility Bot")
st.caption(
    "Ask which IBM / Broadcom products are compatible with a given z/OS release. "
    "Data is scraped from each vendor's **public** compatibility pages — no login required."
)

env_key = os.environ.get("ANTHROPIC_API_KEY")
with st.sidebar:
    st.subheader("🔑 API Key")
    if env_key:
        st.success("Using ANTHROPIC_API_KEY from Secrets/environment.")
        override_key = st.text_input(
            "Override with a different key (optional, this session only)", type="password"
        )
    else:
        st.warning("ANTHROPIC_API_KEY is not set in Secrets/environment.")
        override_key = st.text_input(
            "Paste a key here to test (this session only, not saved)", type="password"
        )
    active_key = override_key.strip() or env_key
    if active_key and not active_key.startswith("sk-ant-"):
        st.error("That doesn't look like an Anthropic key — it should start with `sk-ant-`.")

if not active_key:
    st.warning(
        "No Anthropic API key available yet. Structured lookup still works below; "
        "the written summary needs a key (set it in Streamlit Secrets, or paste one in the sidebar).",
        icon="⚠️",
    )

with st.expander("ℹ️ About this data / why there's no login here"):
    st.markdown(
        """
        - **IBM**: Software Product Compatibility Reports (SPCR) and the
          "detailed system requirements" support pages are public — no IBM ID needed.
        - **Broadcom**: the z/OS compatibility matrices are public pages.
          Broadcom's *full support ticket portal* requires login, but that's
          not needed for compatibility lookups.
        - This app never asks for a username or password. Data shown is
          refreshed by re-running the scrapers and rebuilding the pickle
          (`build_pickle.py`) — see README.md.
        """
    )

st.divider()
col_l, col_r = st.columns([1, 2])

with col_l:
    st.subheader("🔎 Ask the bot")
    zos_version = st.text_input("z/OS version", placeholder="e.g. z/OS V2R5, or 2.5")
    product_name = st.text_input("Product name (optional — leave blank to list all)", placeholder="e.g. Endevor")
    vendor_choice = st.selectbox("Vendor filter", ["Any", "IBM", "Broadcom"])
    ask_btn = st.button("Ask", type="primary", use_container_width=True)

with col_r:
    st.subheader("📚 Known data (for reference)")
    c1, c2 = st.columns(2)
    c1.metric("z/OS versions in DB", len(known_zos_versions))
    c2.metric("Products in DB", len(known_products))
    st.caption("z/OS versions seen: " + ", ".join(known_zos_versions))

st.divider()

if ask_btn:
    if not zos_version.strip():
        st.error("Please enter a z/OS version.")
    else:
        matches = products_compatible_with(df, zos_version, product_filter=product_name or None, vendor_filter=vendor_choice)

        tab_answer, tab_table, tab_chart = st.tabs(["💬 Answer", "📋 Raw matches", "📊 Coverage chart"])

        with tab_answer:
            query = f"{product_name or ''} compatible with z/OS {zos_version}".strip()
            retrieved = retrieve(query, vectorizer, matrix, chunk_texts, chunk_meta, top_k=8)
            if active_key:
                with st.spinner("Asking Claude..."):
                    answer, error = safe_generate_answer(
                        zos_version, matches, retrieved, product_filter=product_name or None, api_key=active_key
                    )
                if error:
                    st.error(error)
                    st.caption("Showing raw matches instead:")
                    st.dataframe(matches, use_container_width=True, hide_index=True)
                else:
                    st.markdown(answer)
            else:
                st.info("Add an API key (sidebar) to get a natural-language summary. Showing raw matches instead:")
                st.dataframe(matches, use_container_width=True, hide_index=True)

        with tab_table:
            if matches.empty:
                st.warning("No compatible products found for this z/OS version / filter in the current data.")
            else:
                st.dataframe(matches, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download results as CSV",
                    matches.to_csv(index=False).encode("utf-8"),
                    file_name=f"compatibility_{zos_version.replace('/', '_')}.csv",
                )

        with tab_chart:
            if not matches.empty:
                fig = px.bar(
                    matches.groupby("vendor").size().reset_index(name="count"),
                    x="vendor", y="count", color="vendor",
                    title=f"Compatible products for {zos_version}, by vendor",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No data to chart.")
else:
    st.info("👈 Enter a z/OS version (and optionally a product) on the left, then click **Ask**.")

st.divider()
st.caption("Data sources are public vendor pages — always re-verify against the live vendor page before production changes.")
