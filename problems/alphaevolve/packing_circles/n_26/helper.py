import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import pdist, squareform


def max_radii(centers: np.ndarray) -> np.ndarray:
    """Largest feasible radii for FIXED centers — the exact optimum, not a heuristic.

    With the centers held fixed the problem is a linear program in the radii:

        maximize    sum(r)
        subject to  r_i + r_j <= ||c_i - c_j||   for every pair i < j
                    0 <= r_i  <= dist(c_i, nearest wall)

    Both constraint families are linear, so the best sum of radii any given
    arrangement can reach is computable exactly. That leaves the search with
    the part that is actually hard — where to put the centers.

    Input: (n, 2) centers. Coordinates are clipped into the unit square first,
    so a center that drifted outside yields a zero radius rather than an
    infeasible program.
    Output: (n,) radii, feasible by construction up to LP tolerance.
    """
    c = np.clip(np.asarray(centers, dtype=float), 0.0, 1.0)
    n = c.shape[0]
    if n == 0:
        return np.zeros(0)

    # Wall clearance: the largest a circle can be before leaving the square.
    upper = np.minimum.reduce([c[:, 0], 1.0 - c[:, 0], c[:, 1], 1.0 - c[:, 1]])
    upper = np.maximum(upper, 0.0)
    if n == 1:
        return upper

    # One row per pair: r_i + r_j <= d_ij.
    iu = np.triu_indices(n, k=1)
    d = squareform(pdist(c))[iu]
    rows = np.zeros((iu[0].size, n))
    rows[np.arange(iu[0].size), iu[0]] = 1.0
    rows[np.arange(iu[1].size), iu[1]] = 1.0

    res = linprog(
        c=-np.ones(n),
        A_ub=rows,
        b_ub=d,
        bounds=list(zip(np.zeros(n), upper)),
        method="highs",
    )
    if not res.success:
        return np.zeros(n)
    # The LP is solved to a tolerance, so nudge off the boundary: validate.py
    # rejects an overlap beyond 1e-6, and a solution sitting exactly on the
    # constraint can land on the wrong side of it.
    return np.maximum(res.x - 1e-9, 0.0)


def pack(centers: np.ndarray) -> np.ndarray:
    """(n, 2) centers -> the (n, 3) array `entrypoint` must return.

    Radii come from :func:`max_radii`, so the returned packing is the best one
    achievable for those centers.
    """
    c = np.clip(np.asarray(centers, dtype=float), 0.0, 1.0)
    return np.column_stack([c, max_radii(c)])


def sum_of_radii(centers: np.ndarray) -> float:
    """Fitness of an arrangement: the objective validate.py scores."""
    return float(np.sum(max_radii(centers)))


def demo() -> None:
    """Self-check: the LP result must satisfy the validator's own constraints."""
    rng = np.random.default_rng(0)
    for n in (1, 2, 26):
        centers = rng.random((n, 2))
        r = max_radii(centers)
        assert r.shape == (n,) and np.all(r >= 0) and np.all(np.isfinite(r))
        # Containment.
        assert np.all(centers[:, 0] - r >= -1e-6) and np.all(
            centers[:, 0] + r <= 1 + 1e-6
        )
        assert np.all(centers[:, 1] - r >= -1e-6) and np.all(
            centers[:, 1] + r <= 1 + 1e-6
        )
        # Non-overlap.
        if n > 1:
            iu = np.triu_indices(n, k=1)
            d = squareform(pdist(centers))[iu]
            assert np.all(r[iu[0]] + r[iu[1]] <= d + 1e-6)
        assert pack(centers).shape == (n, 3)

    # Optimality: 4 circles at the corners of the inscribed square each reach
    # the wall bound, which is the binding constraint there.
    q = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
    assert np.allclose(max_radii(q), 0.25, atol=1e-6)

    # A center outside the square is clipped, not fatal.
    assert np.all(np.isfinite(max_radii(np.array([[-1.0, 2.0], [0.5, 0.5]]))))
    print("helper.py self-check OK")


if __name__ == "__main__":
    demo()
