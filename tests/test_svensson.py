"""Analytic properties of the Svensson curve (eqs. 1, 5, 8, 11, 12)."""
import numpy as np
import pytest

from oatcurve import svensson

# A realistic upward-sloping curve with a hump (decimal units).
# np.trapezoid is NumPy >= 2.0; np.trapz is the older name.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

THETA = (0.035, -0.015, 0.010, -0.008, 1.8, 9.0)


def test_loading_limit_at_zero():
    assert svensson._loading(np.array([0.0]))[0] == pytest.approx(1.0)
    assert svensson._loading(np.array([1e-14]))[0] == pytest.approx(1.0)


def test_short_and_long_end_limits():
    b0, b1 = THETA[0], THETA[1]
    # y(0) = f(0) = beta0 + beta1 (instantaneous short rate)
    assert svensson.zero_yield(0.0, THETA) == pytest.approx(b0 + b1)
    assert svensson.forward_rate(0.0, THETA) == pytest.approx(b0 + b1)
    # y(m) -> beta0 as m -> infinity (long-run level)
    assert svensson.zero_yield(1e4, THETA) == pytest.approx(b0, abs=1e-5)


@pytest.mark.parametrize("m", [0.5, 2.0, 10.0, 30.0])
def test_zero_yield_is_average_forward_rate(m):
    """Eq. (8): y(m) * m equals the integral of f over [0, m]."""
    grid = np.linspace(0.0, m, 20001)
    integral = _trapezoid(svensson.forward_rate(grid, THETA), grid)
    assert svensson.zero_yield(m, THETA) * m == pytest.approx(integral, rel=1e-8)


def test_discount_factor_definition():
    m = np.array([1.0, 5.0, 20.0])
    expected = np.exp(-svensson.zero_yield(m, THETA) * m)
    np.testing.assert_allclose(svensson.discount_factor(m, THETA), expected)


@pytest.mark.parametrize("m", [3.0, 7.0, 10.0, 25.0])
def test_par_bond_prices_at_par(m):
    """Eq. (5): an annual bond paying the par yield as coupon is worth 1."""
    c = svensson.par_yield(m, THETA)
    times = np.arange(1.0, m + 1e-9)
    price = c * svensson.discount_factor(times, THETA).sum() \
        + svensson.discount_factor(m, THETA)
    assert price == pytest.approx(1.0, abs=1e-12)
