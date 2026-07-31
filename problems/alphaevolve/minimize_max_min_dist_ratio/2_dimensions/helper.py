import numpy as np
from scipy.spatial.distance import pdist


def distance_ratio(points: np.ndarray) -> float:
    """The objective validate.py scores: (max pairwise dist / min pairwise dist)^2.

    Works in any dimension. Returns ``inf`` for a degenerate configuration
    (two points closer than the 1e-9 the validator rejects at), so a search
    loop can compare configurations without having to catch an exception.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2 or not np.all(np.isfinite(p)):
        return float("inf")
    d = pdist(p)
    lo = float(np.min(d))
    if lo < 1e-9:
        return float("inf")
    return float((np.max(d) / lo) ** 2)


def extreme_pairs(points: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Which pair sets the maximum distance, and which sets the minimum.

    These two pairs are the only ones the objective depends on, so they are
    the only points worth moving — everything else is slack.
    Returns ((i, j) closest, (k, l) farthest).
    """
    p = np.asarray(points, dtype=float)
    n = p.shape[0]
    d = pdist(p)
    iu = np.triu_indices(n, k=1)
    lo, hi = int(np.argmin(d)), int(np.argmax(d))
    return (int(iu[0][lo]), int(iu[1][lo])), (int(iu[0][hi]), int(iu[1][hi]))


def normalize(points: np.ndarray) -> np.ndarray:
    """Centre at the origin and scale so the minimum pairwise distance is 1.

    The objective is invariant under translation, rotation and uniform
    scaling, so this changes nothing about the fitness — it only keeps
    coordinates in a numerically comfortable range across many optimisation
    steps.
    """
    p = np.asarray(points, dtype=float) - np.mean(points, axis=0)
    d = pdist(p)
    lo = float(np.min(d)) if d.size else 0.0
    return p / lo if lo > 1e-12 else p


def demo() -> None:
    """Self-check: the helper must agree with validate.py on the objective."""
    from validate import validate

    rng = np.random.default_rng(0)
    for shape in ((16, 2), (14, 3)):
        pts = rng.random(shape)
        r = distance_ratio(pts)
        assert np.isfinite(r) and r >= 1.0
        # normalize is a no-op on the objective, by construction.
        assert abs(distance_ratio(normalize(pts)) - r) < 1e-9 * max(r, 1.0)
        (i, j), (k, l) = extreme_pairs(pts)
        d = np.linalg.norm(pts[i] - pts[j]), np.linalg.norm(pts[k] - pts[l])
        assert abs((d[1] / d[0]) ** 2 - r) < 1e-9 * r

    # Same number the validator reports, on the shape this problem expects.
    pts = rng.random((16, 2))
    assert abs(validate(pts)["fitness"] - distance_ratio(pts)) < 1e-9

    # Degenerate input is `inf`, not an exception.
    assert distance_ratio(np.zeros((3, 2))) == float("inf")
    assert distance_ratio(np.array([[np.nan, 0.0], [1.0, 1.0]])) == float("inf")
    print("helper.py self-check OK")


if __name__ == "__main__":
    demo()
