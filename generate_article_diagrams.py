"""Generate SVG figures for the Substack article from real embedding data.

Re-uses the live EmbeddingRouter (real OpenAI embeddings + real centroids),
projects the high-dimensional vectors to 2D via PCA (numpy SVD), and renders
three static SVGs: the embedding space, the centroids, and a sample
classification.
"""
import os

import numpy as np

from llm.embedding_router import get_embedding_router
from prompts.route_examples import ROUTE_EXAMPLES

COLORS = {"small": "#2563eb", "frontier": "#d97706", "specialist": "#059669", "query": "#db2777"}
TIER_LABELS = {"small": "small", "frontier": "frontier", "specialist": "specialist"}

WIDTH, HEIGHT, MARGIN = 640, 380, 50


def pca_2d(vectors: np.ndarray):
    """Project rows of `vectors` onto their top-2 principal components."""
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]  # (2, dim)
    return mean, components


def project(vectors: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (vectors - mean) @ components.T


def to_canvas(points_2d: np.ndarray) -> np.ndarray:
    """Rescale arbitrary 2D points to fit the SVG canvas with a margin."""
    mins = points_2d.min(axis=0)
    maxs = points_2d.max(axis=0)
    span = np.where(maxs - mins == 0, 1, maxs - mins)
    normalized = (points_2d - mins) / span  # 0..1
    xs = MARGIN + normalized[:, 0] * (WIDTH - 2 * MARGIN)
    # flip Y so PCA "up" renders visually up
    ys = HEIGHT - (MARGIN + normalized[:, 1] * (HEIGHT - 2 * MARGIN))
    return np.column_stack([xs, ys])


# ---- tiny SVG-building helpers (string templates, mirroring centroid_explainer.html's drawing primitives) ----

def svg_dot(x, y, color, r=8, opacity=1.0):
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" '
            f'opacity="{opacity}" stroke="#fff" stroke-width="1.5" />')


def svg_ring(x, y, color, r=14, width=3):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="none" stroke="{color}" stroke-width="{width}" />'


def svg_line(x1, y1, x2, y2, color, dash=None, width=1.5, opacity=0.5):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{dash_attr} opacity="{opacity}" />')


def svg_label(x, y, text, color, size=12, weight=700):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="middle" '
            f'font-family="-apple-system, Helvetica, Arial, sans-serif">{text}</text>')


def wrap_svg(body: str, title: str) -> str:
    return (
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="-apple-system, Helvetica, Arial, sans-serif">\n'
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#fafafa" stroke="#e5e7eb" />\n'
        f'<text x="{WIDTH/2}" y="26" text-anchor="middle" font-size="14" font-weight="700" '
        f'fill="#1d1d1f">{title}</text>\n'
        f'{body}\n</svg>\n'
    )


def main():
    router = get_embedding_router()
    routes = list(ROUTE_EXAMPLES.keys())
    flat_texts = [text for route in routes for text in ROUTE_EXAMPLES[route]]
    flat_vectors = np.array(router.embeddings.embed_documents(flat_texts))

    mean, components = pca_2d(flat_vectors)
    projected = project(flat_vectors, mean, components)
    canvas_points = to_canvas(projected)

    # slice canvas points back into per-tier groups, in the same order they were embedded
    groups = {}
    offset = 0
    for route in routes:
        count = len(ROUTE_EXAMPLES[route])
        groups[route] = canvas_points[offset:offset + count]
        offset += count

    # project the *real* centroids (from the live router) into the same PCA basis
    centroid_canvas = {}
    for route in routes:
        centroid_vec = router._route_centroids[route]
        centroid_2d = project(centroid_vec.reshape(1, -1), mean, components)[0]
        centroid_canvas[route] = to_canvas(np.vstack([projected, centroid_2d]))[-1]
        # NOTE: re-running to_canvas with the centroid appended keeps scaling consistent
        # with the example points (same min/max), so position is comparable across figures.

    # --- Figure 1: the embedding space ---
    body1 = []
    for route in routes:
        for (x, y) in groups[route]:
            body1.append(svg_dot(x, y, COLORS[route]))
        cx, cy = groups[route].mean(axis=0)
        body1.append(svg_label(cx, cy - 28, TIER_LABELS[route], COLORS[route]))
    fig1 = wrap_svg(
        "\n".join(body1),
        "Fig 1 — Real reference-example embeddings, projected to 2D via PCA",
    )

    # --- Figure 2: centroids ---
    body2 = []
    for route in routes:
        cx, cy = centroid_canvas[route]
        for (x, y) in groups[route]:
            body2.append(svg_line(x, y, cx, cy, COLORS[route], opacity=0.25))
            body2.append(svg_dot(x, y, COLORS[route], r=6, opacity=0.6))
        body2.append(svg_dot(cx, cy, COLORS[route], r=9))
        body2.append(svg_ring(cx, cy, COLORS[route]))
        body2.append(svg_label(cx, cy - 28, f"{TIER_LABELS[route]} centroid", COLORS[route]))
    fig2 = wrap_svg(
        "\n".join(body2),
        "Fig 2 — Each cluster averaged into its real centroid (router._route_centroids)",
    )

    # --- Figure 3: classify a real sample query ---
    sample_query = "Compare event sourcing vs CQRS for a high-throughput trading platform and recommend one with justification"
    route, score = router.classify(sample_query)
    query_vector = np.array(router.embeddings.embed_query(sample_query))
    query_2d = project(query_vector.reshape(1, -1), mean, components)[0]
    query_canvas = to_canvas(np.vstack([projected, query_2d]))[-1]
    qx, qy = query_canvas

    scores = {
        r: router._cosine_similarity(query_vector, router._route_centroids[r])
        for r in routes
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    winner = ranked[0][0]

    body3 = []
    for r in routes:
        for (x, y) in groups[r]:
            body3.append(svg_dot(x, y, COLORS[r], r=5, opacity=0.25))
    for r in routes:
        cx, cy = centroid_canvas[r]
        body3.append(svg_dot(cx, cy, COLORS[r], r=9))
        ring_width = 5 if r == winner else 3
        ring_radius = 17 if r == winner else 14
        body3.append(svg_ring(cx, cy, COLORS[r], r=ring_radius, width=ring_width))
        body3.append(svg_label(cx, cy - 28, TIER_LABELS[r], COLORS[r]))
        body3.append(svg_line(qx, qy, cx, cy, COLORS[r], dash="5,4", opacity=0.6))
    body3.append(svg_dot(qx, qy, COLORS["query"], r=10))
    body3.append(svg_label(qx, qy - 16, "? query", COLORS["query"]))

    score_lines = "  |  ".join(f"{TIER_LABELS[r]}={s:.2f}" for r, s in ranked)
    body3.append(svg_label(WIDTH / 2, HEIGHT - 16, score_lines, "#374151", size=12, weight=600))

    fig3 = wrap_svg(
        "\n".join(body3),
        f'Fig 3 — Real query routed to "{winner}" (score={score:.2f}) via cosine similarity',
    )

    os.makedirs("assets", exist_ok=True)
    for filename, svg in [
        ("fig1_embedding_space.svg", fig1),
        ("fig2_centroids.svg", fig2),
        ("fig3_classification.svg", fig3),
    ]:
        path = os.path.join("assets", filename)
        with open(path, "w") as f:
            f.write(svg)
        print(f"wrote {path}")

    print(f"\nSample query: {sample_query!r}")
    print(f"Routed to: {winner}  (scores: {scores})")


if __name__ == "__main__":
    main()
