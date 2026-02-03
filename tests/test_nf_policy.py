import jax
import jax.numpy as jnp

from algorithms.nf_policy import NFActor


def _make_inputs(key, batch_size, state_dim, action_dim, chunk_len, rtc_prefix_len):
    key, k1, k2 = jax.random.split(key, 3)
    states = jax.random.normal(k1, (batch_size, state_dim))
    prev_actions = jax.random.normal(k2, (batch_size, rtc_prefix_len, action_dim))
    return key, states, prev_actions


def test_nf_actor_sample_shape():
    key = jax.random.PRNGKey(0)
    batch_size = 3
    state_dim = 11
    rtc_prefix_len = 2
    action_dim = 4
    chunk_len = 5

    key, states, prev_actions = _make_inputs(
        key, batch_size, state_dim, action_dim, chunk_len, rtc_prefix_len
    )
    model = NFActor(
        action_dim=action_dim,
        chunk_len=chunk_len,
        hidden_dim=64,
        n_hiddens=2,
        num_layers=2,
        scale_max=1.0,
    )

    key, init_key, sample_key = jax.random.split(key, 3)
    params = model.init(
        {"params": init_key, "dropout": sample_key},
        states,
        prev_actions,
        rng=sample_key,
        train=True,
        method=NFActor.sample,
    )
    actions = model.apply(
        params,
        states,
        prev_actions,
        rng=sample_key,
        train=False,
        method=NFActor.sample,
    )

    assert actions.shape == (batch_size, chunk_len, action_dim)


def test_nf_actor_log_prob_shape_and_finite():
    key = jax.random.PRNGKey(1)
    batch_size = 4
    state_dim = 9
    rtc_prefix_len = 1
    action_dim = 3
    chunk_len = 6

    key, states, prev_actions = _make_inputs(
        key, batch_size, state_dim, action_dim, chunk_len, rtc_prefix_len
    )
    model = NFActor(
        action_dim=action_dim,
        chunk_len=chunk_len,
        hidden_dim=64,
        n_hiddens=2,
        num_layers=2,
        scale_max=1.0,
    )

    key, init_key, sample_key = jax.random.split(key, 3)
    params = model.init(
        {"params": init_key, "dropout": sample_key},
        states,
        prev_actions,
        rng=sample_key,
        train=True,
        method=NFActor.sample,
    )
    actions = model.apply(
        params,
        states,
        prev_actions,
        rng=sample_key,
        train=False,
        method=NFActor.sample,
    )
    logp = model.apply(params, actions, states, prev_actions, train=False, method=NFActor.log_prob)

    assert logp.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(logp))


def test_nf_actor_outputs_bounded_actions():
    key = jax.random.PRNGKey(2)
    batch_size = 5
    state_dim = 7
    rtc_prefix_len = 3
    action_dim = 2
    chunk_len = 4

    key, states, prev_actions = _make_inputs(
        key, batch_size, state_dim, action_dim, chunk_len, rtc_prefix_len
    )
    model = NFActor(
        action_dim=action_dim,
        chunk_len=chunk_len,
        hidden_dim=64,
        n_hiddens=2,
        num_layers=2,
        scale_max=1.0,
    )

    key, init_key, sample_key = jax.random.split(key, 3)
    params = model.init(
        {"params": init_key, "dropout": sample_key},
        states,
        prev_actions,
        rng=sample_key,
        train=True,
        method=NFActor.sample,
    )
    actions = model.apply(
        params,
        states,
        prev_actions,
        rng=sample_key,
        train=False,
        method=NFActor.sample,
    )

    assert jnp.all(actions <= 1.0 + 1e-6)
    assert jnp.all(actions >= -1.0 - 1e-6)


def test_nf_actor_sample_n_shape_and_finite():
    key = jax.random.PRNGKey(3)
    batch_size = 2
    state_dim = 5
    rtc_prefix_len = 1
    action_dim = 3
    chunk_len = 4
    num_samples = 6

    key, states, prev_actions = _make_inputs(
        key, batch_size, state_dim, action_dim, chunk_len, rtc_prefix_len
    )
    model = NFActor(
        action_dim=action_dim,
        chunk_len=chunk_len,
        hidden_dim=64,
        n_hiddens=2,
        num_layers=2,
        scale_max=1.0,
    )

    key, init_key, sample_key = jax.random.split(key, 3)
    params = model.init(
        {"params": init_key, "dropout": sample_key},
        states,
        prev_actions,
        rng=sample_key,
        train=True,
        method=NFActor.sample,
    )
    actions = model.apply(
        params,
        states,
        prev_actions,
        rng=sample_key,
        num_samples=num_samples,
        z_scale=0.7,
        z_clip=1.5,
        train=False,
        method=NFActor.sample_n,
    )

    assert actions.shape == (batch_size, num_samples, chunk_len, action_dim)
    assert jnp.all(jnp.isfinite(actions))


def test_nf_actor_uniform_base_log_prob_finite():
    key = jax.random.PRNGKey(4)
    batch_size = 3
    state_dim = 6
    rtc_prefix_len = 2
    action_dim = 2
    chunk_len = 3

    key, states, prev_actions = _make_inputs(
        key, batch_size, state_dim, action_dim, chunk_len, rtc_prefix_len
    )
    model = NFActor(
        action_dim=action_dim,
        chunk_len=chunk_len,
        hidden_dim=64,
        n_hiddens=2,
        num_layers=2,
        scale_max=1.0,
        base_dist="uniform",
    )

    key, init_key, sample_key = jax.random.split(key, 3)
    params = model.init(
        {"params": init_key, "dropout": sample_key},
        states,
        prev_actions,
        rng=sample_key,
        train=True,
        method=NFActor.sample,
    )
    actions = model.apply(
        params,
        states,
        prev_actions,
        rng=sample_key,
        train=False,
        method=NFActor.sample,
    )
    logp = model.apply(params, actions, states, prev_actions, train=False, method=NFActor.log_prob)

    assert logp.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(logp))


def test_nf_actor_uniform_identity_log_prob_constant():
    key = jax.random.PRNGKey(5)
    batch_size = 6
    state_dim = 4
    rtc_prefix_len = 1
    action_dim = 2
    chunk_len = 3

    key, states, prev_actions = _make_inputs(
        key, batch_size, state_dim, action_dim, chunk_len, rtc_prefix_len
    )
    model = NFActor(
        action_dim=action_dim,
        chunk_len=chunk_len,
        hidden_dim=32,
        n_hiddens=2,
        num_layers=0,
        scale_max=1.0,
        base_dist="uniform",
        use_plu=False,
    )

    key, init_key, sample_key = jax.random.split(key, 3)
    params = model.init(
        {"params": init_key, "dropout": sample_key},
        states,
        prev_actions,
        rng=sample_key,
        train=True,
        method=NFActor.sample,
    )

    actions = jax.random.uniform(sample_key, (batch_size, chunk_len, action_dim), minval=-0.9, maxval=0.9)
    logp = model.apply(params, actions, states, prev_actions, method=NFActor.log_prob)

    max_diff = jnp.max(jnp.abs(logp - logp[0]))
    assert max_diff < 1e-5
