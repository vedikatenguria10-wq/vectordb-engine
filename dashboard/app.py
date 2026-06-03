"""Streamlit benchmark dashboard for VectorDB Engine.

Run from project root:
  streamlit run dashboard/app.py --server.port 8501
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hybrid_search import BM25Search, HybridSearch
from dashboard.charts import (
    hybrid_comparison_chart,
    latency_bar_chart,
    pca_scatter_plot,
)
from db.vector_store import VectorStore
from dotenv import load_dotenv
from embeddings.encoder import SentenceEncoder

API_BASE = "http://localhost:8080"


@st.cache_resource
def _vector_store() -> VectorStore:
    """Cached VectorStore using DB_PATH from the environment."""
    load_dotenv()
    db_path = os.getenv("DB_PATH", "./data/vectordb")
    return VectorStore(db_path)


@st.cache_resource
def _encoder() -> SentenceEncoder:
    """Cached sentence encoder for query highlighting and hybrid search."""
    return SentenceEncoder()


def _build_hybrid_search(store: VectorStore) -> HybridSearch:
    """Build a BM25 + hybrid searcher from the current vector store contents."""
    items = store.list_all()
    documents = [str(item["text"]) for item in items]
    doc_ids = [int(item["id"]) for item in items]
    bm25 = BM25Search()
    bm25.build_index(documents, doc_ids)
    return HybridSearch(store, bm25)


def _fetch_benchmark(query: str, k: int) -> dict[str, Any]:
    """Call GET /api/benchmark on the running API server."""
    response = requests.get(
        f"{API_BASE}/api/benchmark",
        params={"q": query, "k": k, "metric": "cosine"},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def _fetch_stats() -> dict[str, Any]:
    """Call GET /api/stats on the running API server."""
    response = requests.get(f"{API_BASE}/api/stats", timeout=30)
    response.raise_for_status()
    return response.json()


tab_benchmark, tab_semantic, tab_hybrid = st.tabs(
    ["Algorithm Benchmark", "Semantic Space", "Hybrid vs Semantic"]
)

with tab_benchmark:
    st.header("Algorithm Benchmark")
    bench_query = st.text_input("Search query", value="technology news", key="bench_q")
    bench_k = st.slider("Top k", min_value=1, max_value=20, value=5, key="bench_k")

    if st.button("Run Benchmark", type="primary"):
        try:
            with st.spinner("Running all algorithms…"):
                benchmark = _fetch_benchmark(bench_query, bench_k)

            timing_input = {
                algo: {"time_ms": float(data["time_ms"])}
                for algo, data in benchmark.items()
            }
            st.plotly_chart(latency_bar_chart(timing_input), use_container_width=True)

            cols = st.columns(len(benchmark))
            for col, (algo, data) in zip(cols, benchmark.items()):
                with col:
                    st.subheader(algo)
                    st.caption(f"{data['time_ms']:.2f} ms")
                    results = data.get("results", [])
                    if results:
                        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                    else:
                        st.info("No results")
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
            st.caption(f"Ensure the API is running at {API_BASE}")

with tab_semantic:
    st.header("Semantic Space")
    try:
        stats = _fetch_stats()
        vector_stats = stats.get("vector_store", {})
        st.metric("Vectors in store", int(vector_stats.get("count", 0)))
        st.caption(
            f"Dimensions: {vector_stats.get('dims', 0)} | "
            f"Chunks: {stats.get('document_store', {}).get('chunk_count', 0)}"
        )

        store = _vector_store()
        items = store.list_all()
        if not items:
            st.warning("No vectors in VectorStore. Run scripts/load_dataset.py first.")
        else:
            vectors = [np.asarray(item["vector"], dtype=np.float32) for item in items]
            labels = [str(item["text"])[:80] for item in items]
            categories = [str(item["category"]) for item in items]

            highlight_q = st.text_input(
                "Highlight query on plot (optional)",
                value="",
                key="pca_q",
            )
            query_vector = None
            if highlight_q.strip():
                query_vector = _encoder().encode(highlight_q.strip())

            fig = pca_scatter_plot(
                vectors,
                labels,
                categories,
                query_vector=query_vector,
                query_label=highlight_q.strip() or "Query",
            )
            st.plotly_chart(fig, use_container_width=True)
    except requests.RequestException as exc:
        st.error(f"Could not reach API stats: {exc}")
    except Exception as exc:
        st.error(f"Failed to load semantic space: {exc}")

with tab_hybrid:
    st.header("Hybrid vs Semantic")
    hybrid_query = st.text_input("Search query", value="business markets", key="hybrid_q")
    hybrid_k = st.slider("Top k", min_value=1, max_value=20, value=5, key="hybrid_k")
    alpha = st.slider(
        "Alpha (semantic vs keyword weight)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="hybrid_alpha",
    )
    st.caption(
        "**Alpha** controls reciprocal-rank fusion: higher alpha favors semantic (vector) "
        "ranking; lower alpha favors BM25 keyword ranking. "
        "`combined = alpha × 1/(60+semantic_rank) + (1-alpha) × 1/(60+bm25_rank)`"
    )

    if st.button("Compare", type="primary"):
        try:
            store = _vector_store()
            encoder = _encoder()
            query_vector = encoder.encode(hybrid_query)

            with st.spinner("Running semantic and hybrid search…"):
                semantic_raw = store.search(query_vector, hybrid_k)
                semantic_results = [
                    {
                        "id": int(hit["id"]),
                        "text": str(hit["text"]),
                        "category": str(hit["category"]),
                        "score": float(hit.get("score", 0.0)),
                    }
                    for hit in semantic_raw
                ]

                hybrid = _build_hybrid_search(store)
                hybrid_results = hybrid.search(hybrid_query, hybrid_k, alpha=alpha)

            st.plotly_chart(
                hybrid_comparison_chart(semantic_results, hybrid_results),
                use_container_width=True,
            )

            left, right = st.columns(2)
            with left:
                st.subheader("Semantic results")
                st.dataframe(pd.DataFrame(semantic_results), use_container_width=True, hide_index=True)
            with right:
                st.subheader("Hybrid results")
                st.dataframe(pd.DataFrame(hybrid_results), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")

st.sidebar.title("VectorDB Dashboard")
st.sidebar.markdown(f"API: `{API_BASE}`")
st.sidebar.markdown("Port: **8501**")
st.sidebar.caption("Start API: `python main.py`")
