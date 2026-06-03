"""Plotly chart builders for the VectorDB benchmark dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_ALGO_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
]


def _pca_2d(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project rows to 2D with PCA via SVD; returns coords, mean, components."""
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    if centered.shape[0] < 2:
        coords = np.zeros((centered.shape[0], 2), dtype=np.float32)
        return coords, mean, np.eye(matrix.shape[1], 2, dtype=np.float32)

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    coords = centered @ components
    return coords, mean, components


def latency_bar_chart(benchmark_results: dict[str, dict[str, float]]) -> go.Figure:
    """Build a color-coded bar chart of latency (ms) per algorithm."""
    algos = list(benchmark_results.keys())
    times = [float(benchmark_results[algo]["time_ms"]) for algo in algos]
    colors = [_ALGO_COLORS[i % len(_ALGO_COLORS)] for i in range(len(algos))]

    fig = go.Figure(
        data=[
            go.Bar(
                x=algos,
                y=times,
                marker_color=colors,
                text=[f"{t:.2f} ms" for t in times],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Search latency by algorithm",
        xaxis_title="Algorithm",
        yaxis_title="Time (ms)",
        template="plotly_white",
    )
    return fig


def pca_scatter_plot(
    vectors: list[np.ndarray],
    labels: list[str],
    categories: list[str],
    query_vector: np.ndarray | None = None,
    query_label: str = "Query",
) -> go.Figure:
    """Scatter plot of vectors projected to 2D with PCA, colored by category."""
    if not vectors:
        fig = go.Figure()
        fig.update_layout(title="No vectors to display", template="plotly_white")
        return fig

    matrix = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors])
    coords, mean, components = _pca_2d(matrix)
    unique_categories = sorted(set(categories))
    category_colors = {
        cat: _ALGO_COLORS[i % len(_ALGO_COLORS)]
        for i, cat in enumerate(unique_categories)
    }

    fig = go.Figure()
    for cat in unique_categories:
        mask = [c == cat for c in categories]
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=cat,
                marker=dict(size=9, color=category_colors[cat], opacity=0.75),
                text=[labels[i] for i, m in enumerate(mask) if m],
                hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            )
        )

    if query_vector is not None:
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        q_coord = (q - mean) @ components
        fig.add_trace(
            go.Scatter(
                x=[float(q_coord[0])],
                y=[float(q_coord[1])],
                mode="markers",
                name=query_label,
                marker=dict(size=14, color="#FF0000", symbol="star", line=dict(width=1)),
                text=[query_label],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Semantic space (PCA 2D)",
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_white",
        legend_title="Category",
    )
    return fig


def hybrid_comparison_chart(
    semantic_results: list[dict[str, Any]],
    hybrid_results: list[dict[str, Any]],
) -> go.Figure:
    """Side-by-side comparison of semantic vs hybrid ranking positions."""
    sem_ranks = {int(r["id"]): i + 1 for i, r in enumerate(semantic_results)}
    hyb_ranks = {int(r["id"]): i + 1 for i, r in enumerate(hybrid_results)}

    def _label(result: dict[str, Any]) -> str:
        text = str(result.get("text", ""))
        return text[:40] + ("…" if len(text) > 40 else "")

    sem_labels = [_label(r) for r in semantic_results]
    hyb_labels = [_label(r) for r in hybrid_results]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Semantic (vector)", "Hybrid (RRF)"),
        horizontal_spacing=0.12,
    )

    if semantic_results:
        fig.add_trace(
            go.Bar(
                x=sem_ranks.values(),
                y=sem_labels,
                orientation="h",
                marker_color="#636EFA",
                name="Semantic rank",
                text=[f"#{r}" for r in sem_ranks.values()],
                textposition="outside",
            ),
            row=1,
            col=1,
        )

    if hybrid_results:
        fig.add_trace(
            go.Bar(
                x=hyb_ranks.values(),
                y=hyb_labels,
                orientation="h",
                marker_color="#00CC96",
                name="Hybrid rank",
                text=[f"#{r}" for r in hyb_ranks.values()],
                textposition="outside",
            ),
            row=1,
            col=2,
        )

    all_ids = set(sem_ranks) | set(hyb_ranks)
    changes: list[str] = []
    for doc_id in all_ids:
        sr = sem_ranks.get(doc_id)
        hr = hyb_ranks.get(doc_id)
        if sr is None and hr is not None:
            changes.append(f"id {doc_id}: new in hybrid @ #{hr}")
        elif hr is None and sr is not None:
            changes.append(f"id {doc_id}: dropped from hybrid (was #{sr})")
        elif sr is not None and hr is not None and sr != hr:
            delta = hr - sr
            changes.append(f"id {doc_id}: #{sr} → #{hr} ({delta:+d})")

    fig.update_xaxes(title_text="Rank position", row=1, col=1, autorange="reversed")
    fig.update_xaxes(title_text="Rank position", row=1, col=2, autorange="reversed")
    fig.update_layout(
        title="Semantic vs hybrid rank comparison",
        template="plotly_white",
        height=max(400, 80 * max(len(semantic_results), len(hybrid_results), 1)),
        showlegend=False,
    )

    if changes:
        fig.add_annotation(
            text="Rank changes: " + " | ".join(changes[:8])
            + (" …" if len(changes) > 8 else ""),
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.18,
            showarrow=False,
            font=dict(size=11),
        )

    return fig
