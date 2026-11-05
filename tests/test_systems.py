import numpy as np
import pytest

from filter_tester.systems import (
    AnalyticSystem,
    DiscreteSystem,
    GaussianNoise,
    SDESystem,
    UniformNoise,
    simulate,
)


# --------------------------------------------------------------------- noise
def test_gaussian_noise_moments():
    cov = np.array([[1.0, 0.3], [0.3, 0.5]])
    noise = GaussianNoise(cov, mean=[1.0, -2.0])
    rng = np.random.default_rng(0)
    samples = np.array([noise(rng) for _ in range(20000)])
    assert np.allclose(samples.mean(axis=0), [1.0, -2.0], atol=0.03)
    assert np.allclose(np.cov(samples.T), cov, atol=0.03)


@pytest.mark.parametrize(
    "cov, mean",
    [([[1.0, 0.0]], None), ([[1.0, 0.5], [0.0, 1.0]], None), ([[1.0]], [0.0, 0.0])],
)
def test_gaussian_noise_rejects_bad_arguments(cov, mean):
    with pytest.raises(ValueError):
        GaussianNoise(cov, mean)


def test_uniform_noise_bounds():
    noise = UniformNoise([-1.0, 0.0], [1.0, 0.5])
    rng = np.random.default_rng(1)
    samples = np.array([noise(rng) for _ in range(1000)])
    assert np.all(samples >= [-1.0, 0.0]) and np.all(samples <= [1.0, 0.5])


# ------------------------------------------------------------- base behaviour
def make_random_walk():
    return DiscreteSystem(lambda x, w, t, dt: x + w, x0=[0.0], noise=GaussianNoise([[1.0]]))


def test_advance_requires_reset():
    with pytest.raises(RuntimeError):
        make_random_walk().advance(1.0)


def test_time_must_not_decrease():
    system = make_random_walk()
    system.reset(0.0, rng=0)
    system.advance(1.0)
    with pytest.raises(ValueError):
        system.advance(0.5)


def test_same_time_returns_same_state():
    system = make_random_walk()
    system.reset(0.0, rng=0)
    x1 = system.advance(1.0)
    assert np.array_equal(system.advance(1.0), x1)
    assert system.t == 1.0 and np.array_equal(system.state, x1)


def test_seed_makes_runs_reproducible():
    times = np.arange(10.0)
    a = simulate(make_random_walk(), times, rng=123)
    b = simulate(make_random_walk(), times, rng=123)
    c = simulate(make_random_walk(), times, rng=124)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_returned_state_is_a_copy():
    system = make_random_walk()
    x = system.reset(0.0, rng=0)
    x[0] = 99.0
    assert system.state[0] == 0.0


# ------------------------------------------------------------------- analytic
def test_analytic_deterministic():
    system = AnalyticSystem(lambda t, w: [np.sin(t), np.cos(t)])
    times = np.linspace(0, 3, 7)
    states = simulate(system, times)
    assert np.allclose(states, np.column_stack([np.sin(times), np.cos(times)]))


def test_analytic_non_additive_noise():
    # Noise on the phase, not added to the output.
    system = AnalyticSystem(lambda t, w: [np.sin(t + w[0])], GaussianNoise([[0.01]]))
    states = simulate(system, np.linspace(0, 10, 50), rng=0)
    assert np.all(np.abs(states) <= 1.0)
    assert not np.allclose(states[:, 0], np.sin(np.linspace(0, 10, 50)))


# ------------------------------------------------------------------- discrete
def test_discrete_linear_deterministic():
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    system = DiscreteSystem(lambda x, w, t, dt: A @ x, x0=[0.0, 1.0])
    states = simulate(system, np.arange(0, 1.05, 0.1))
    assert np.allclose(states[-1], [1.0, 1.0])


def test_discrete_receives_variable_dt():
    seen = []

    def f(x, w, t, dt):
        seen.append((t, dt))
        return x

    simulate(DiscreteSystem(f, x0=[0.0]), [0.0, 0.1, 0.4, 1.0])
    assert np.allclose(seen, [(0.0, 0.1), (0.1, 0.3), (0.4, 0.6)])


def test_discrete_random_walk_variance():
    finals = [simulate(make_random_walk(), np.arange(11.0), rng=s)[-1, 0] for s in range(4000)]
    assert abs(np.var(finals) - 10.0) < 0.6


def test_discrete_random_initial_state():
    system = DiscreteSystem(lambda x, w, t, dt: x, x0=GaussianNoise(np.eye(2)))
    x0s = np.array([system.reset(rng=s) for s in range(3000)])
    assert np.allclose(x0s.mean(axis=0), 0.0, atol=0.06)
    assert np.allclose(np.cov(x0s.T), np.eye(2), atol=0.08)


def test_discrete_rejects_shape_change():
    system = DiscreteSystem(lambda x, w, t, dt: np.append(x, 0.0), x0=[0.0])
    system.reset()
    with pytest.raises(ValueError):
        system.advance(1.0)


# --------------------------------------------------------------------- SDE
def test_sde_deterministic_converges_to_exact_solution():
    system = SDESystem(f=lambda x, t: -x, x0=[1.0], max_step=1e-4)
    system.reset(0.0)
    assert np.isclose(system.advance(1.0)[0], np.exp(-1.0), rtol=1e-3)


def test_sde_ornstein_uhlenbeck_moments():
    theta, sigma, x0, T = 1.5, 0.8, 2.0, 1.0
    system = SDESystem(f=lambda x, t: -theta * x, G=lambda x, t: [sigma], x0=[x0], max_step=1e-2)
    finals = np.empty(4000)
    for s in range(finals.size):
        system.reset(0.0, rng=s)
        finals[s] = system.advance(T)[0]
    mean = x0 * np.exp(-theta * T)
    var = sigma**2 / (2 * theta) * (1 - np.exp(-2 * theta * T))
    assert abs(finals.mean() - mean) < 0.03
    assert abs(finals.var() - var) / var < 0.08


def test_sde_matrix_diffusion_correlates_components():
    # Both components driven by the same 1-D Wiener process.
    system = SDESystem(f=lambda x, t: [0.0, 0.0], G=lambda x, t: [[1.0], [1.0]], x0=[0.0, 0.0])
    system.reset(rng=0)
    x = system.advance(1.0)
    assert np.isclose(x[0], x[1])


def test_sde_substep_count_independent_of_call_spacing():
    # Calling advance() often or rarely uses the same step size h.
    calls = []
    system = SDESystem(f=lambda x, t: calls.append(t) or [0.0], x0=[0.0], max_step=0.1)
    simulate(system, [0.0, 1.0])
    coarse = len(calls)
    calls.clear()
    simulate(system, np.linspace(0, 1, 11))
    assert coarse == len(calls) == 10


def test_sde_rejects_bad_diffusion_shape():
    system = SDESystem(f=lambda x, t: [0.0, 0.0], G=lambda x, t: np.ones((3, 1)), x0=[0.0, 0.0])
    system.reset()
    with pytest.raises(ValueError):
        system.advance(0.1)


# ------------------------------------------------------------------ simulate
def test_simulate_validates_times():
    with pytest.raises(ValueError):
        simulate(make_random_walk(), [])
    with pytest.raises(ValueError):
        simulate(make_random_walk(), [0.0, 2.0, 1.0])


# ------------------------------------------------------------ initial state
def test_initial_state_can_be_chosen_per_run():
    system = make_random_walk()
    system.noise_enabled = False
    assert np.array_equal(simulate(system, [0.0, 1.0], x0=[5.0])[:, 0], [5.0, 5.0])
    # Without x0 the system's own initial state is used again.
    assert simulate(system, [0.0, 1.0])[0, 0] == 0.0


def test_initial_state_override_can_be_random():
    system = make_random_walk()
    x0s = [system.reset(rng=s, x0=GaussianNoise([[4.0]]))[0] for s in range(3000)]
    assert abs(np.var(x0s) - 4.0) < 0.3


def test_analytic_system_rejects_initial_state():
    with pytest.raises(TypeError):
        AnalyticSystem(lambda t, w: [t]).reset(x0=[1.0])


def test_heun_is_second_order_for_the_drift():
    def final(method, step):
        s = SDESystem(f=lambda x, t: np.array([x[1], -np.sin(x[0])]), x0=[2.0, 0.0], max_step=step, method=method)
        s.reset()
        return s.advance(3.0)

    reference = final("heun", 1e-4)
    err = {m: [np.abs(final(m, h) - reference).max() for h in (0.02, 0.01)] for m in ("euler", "heun")}
    assert err["euler"][0] / err["euler"][1] == pytest.approx(2, rel=0.25)  # first order
    assert err["heun"][0] / err["heun"][1] == pytest.approx(4, rel=0.25)  # second order
    assert err["heun"][1] < err["euler"][1] / 50


def test_heun_keeps_the_ito_noise():
    theta, sigma, T = 1.5, 0.8, 1.0
    system = SDESystem(f=lambda x, t: -theta * x, G=lambda x, t: [sigma], x0=[2.0], max_step=1e-2, method="heun")
    finals = np.array([(system.reset(0.0, rng=s), system.advance(T))[1][0] for s in range(4000)])
    assert abs(finals.mean() - 2.0 * np.exp(-theta * T)) < 0.03
    var = sigma**2 / (2 * theta) * (1 - np.exp(-2 * theta * T))
    assert abs(finals.var() - var) / var < 0.08


def test_sde_method_validation():
    with pytest.raises(ValueError):
        SDESystem(f=lambda x, t: x, x0=[0.0], method="rk4")
