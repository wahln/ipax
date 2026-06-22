"""Unit tests for fixed/degenerate bound relaxation at initialization."""

from __future__ import annotations

from ipax.ipm.init import relax_fixed_bounds
from tests._helpers import array


def test_fixed_bound_pair_is_widened_symmetrically(namespace):
    # x1 is fixed (l == u == 2); x0 is a normal box [0, 10].
    lower = array(namespace, [0.0, 2.0])
    upper = array(namespace, [10.0, 2.0])
    new_l, new_u = relax_fixed_bounds(namespace, lower, upper)

    # The fixed pair now has a strict interior around its midpoint...
    assert float(new_l[1]) < 2.0 < float(new_u[1])
    assert float(new_u[1]) - float(new_l[1]) > 0.0
    # ...and the well-separated box is untouched.
    assert float(new_l[0]) == 0.0
    assert float(new_u[0]) == 10.0


def test_one_sided_and_absent_bounds_are_untouched(namespace):
    inf = float("inf")
    lower = array(namespace, [0.0, -inf])
    upper = array(namespace, [inf, 5.0])
    new_l, new_u = relax_fixed_bounds(namespace, lower, upper)
    assert float(new_l[0]) == 0.0 and float(new_u[1]) == 5.0
    assert not bool(namespace.isfinite(new_u[0]))
    assert not bool(namespace.isfinite(new_l[1]))

    assert relax_fixed_bounds(namespace, None, None) == (None, None)
