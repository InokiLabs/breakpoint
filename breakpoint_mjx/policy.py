"""Train, checkpoint, save, and reload a frozen locomotion policy.

Robot-agnostic via the Target abstraction (Upgrade #1): env loading, training, and
make_policy_fn all take a Target (default: go1). The network is rebuilt from
brax_ppo_config(env_name) + env observation/action sizes, so any Playground
locomotion env trained with its default PPO config generalizes -- no separate
network config needs storing.

All heavyweight work (actual training) happens on Colab; this module is authored
here and executed there.
"""
from __future__ import annotations

import functools
import pickle
import time
from pathlib import Path

import jax
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

from breakpoint_mjx.targets import get_target

ENV_NAME = "Go1JoystickFlatTerrain"
IMPL = "jax"


def _resolve_target(target):
    """Default to the go1 target so existing call sites keep working."""
    return target if target is not None else get_target("go1")


def load_env(env_name=ENV_NAME):
    """Load a Playground locomotion env, forcing the MJX JAX backend."""
    return registry.load(env_name, config_overrides={"impl": IMPL})


def get_ppo_params(env_name=ENV_NAME, domain="locomotion"):
    """Return the Playground-recommended PPO ConfigDict for env_name.

    T7: ``domain`` selects the suite's param module -- "locomotion" (default; go1/barkour/
    g1) uses ``locomotion_params``; "manipulation" (panda) uses ``manipulation_params``
    (imported lazily so a CPU-only import of this module does not require it).
    """
    if domain == "manipulation":
        from mujoco_playground.config import manipulation_params
        return manipulation_params.brax_ppo_config(env_name, IMPL)
    return locomotion_params.brax_ppo_config(env_name, IMPL)


def _network_factory(pp):
    """Wrap pp.network_factory dict into a partial suitable for ppo.train."""
    return functools.partial(ppo_networks.make_ppo_networks, **dict(pp.network_factory))


def train_policy(out_dir, target=None, num_timesteps=None, num_envs=None, seed=0):
    """Train the target's env with Playground's PPO config + domain randomization.

    Checkpoints stream to <out_dir>/checkpoints (survives Colab disconnects);
    final params are pickled to <out_dir>/<target.checkpoint_name>.
    Returns (make_inference_fn, params).
    """
    tgt = _resolve_target(target)
    env_name = tgt.env_name
    env, eval_env = tgt.load_env(), tgt.load_env()
    pp = get_ppo_params(env_name, getattr(tgt, "domain", "locomotion"))
    nf = _network_factory(pp)

    # Convert ConfigDict to plain dict and strip network_factory (passed separately).
    training_params = dict(pp)
    del training_params["network_factory"]

    if num_timesteps is not None:
        training_params["num_timesteps"] = int(num_timesteps)
    if num_envs is not None:
        training_params["num_envs"] = int(num_envs)

    ckpt = Path(out_dir) / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _first = [True]

    def progress(step, metrics):
        if _first[0]:
            print("metric keys:", sorted(metrics.keys()), flush=True)
            _first[0] = False
        # brax PPO has used both "eval/episode_reward" and ".../mean" across versions.
        r = metrics.get("eval/episode_reward", metrics.get("eval/episode_reward/mean"))
        r = float(r) if r is not None else float("nan")
        print(
            f"[{time.time()-t0:7.0f}s] step={int(step):>12,} eval_reward={r:.3f}",
            flush=True,
        )

    # Not every Playground env registers a domain randomizer (e.g. BarkourJoystick,
    # Spot*). Across Playground versions a missing one may return None or raise; either
    # way fall back to randomization_fn=None so the policy trains without DR (fine for a
    # frozen system-under-test) instead of crashing.
    try:
        randomization_fn = registry.get_domain_randomizer(env_name)
    except (KeyError, ValueError):
        randomization_fn = None

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=nf,
        randomization_fn=randomization_fn,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        save_checkpoint_path=str(ckpt),
        seed=seed,
    )

    make_inference_fn, params, _ = train_fn(
        environment=env, eval_env=eval_env, progress_fn=progress
    )

    save_params(out_dir, params, checkpoint_name=tgt.checkpoint_name)
    print("saved frozen policy to", out_dir, flush=True)
    return make_inference_fn, params




def save_params(out_dir, params, checkpoint_name="go1_policy.pkl"):
    """Pickle params (moved to host) under <out_dir>/<checkpoint_name>."""
    p = Path(out_dir) / checkpoint_name
    p.write_bytes(pickle.dumps(jax.device_get(params)))
    return str(p)


def load_params(out_dir, checkpoint_name="go1_policy.pkl"):
    """Load pickled params from <out_dir>/<checkpoint_name>."""
    return pickle.loads((Path(out_dir) / checkpoint_name).read_bytes())


# VERIFY: if make_ppo_networks rejects a dict observation_size (env.observation_size
# is {"state": (48,), "privileged_state": (123,)}) or does not accept
# preprocess_observations_fn, inspect:
#   import inspect; print(inspect.signature(ppo_networks.make_ppo_networks))
# and adjust accordingly.  The contract for this function is that it returns a
# callable  policy_fn(obs_dict, key) -> (action, extras)  where obs_dict has the
# same structure as state.obs produced by the env.
def make_policy_factory(target=None, deterministic=True):
    """Return ``make_policy(params) -> policy_fn(obs, key)``, params UNBOUND.

    ``make_policy_fn`` bakes params into a closure, so every policy version builds a
    new callable, which makes a new jit cache key and forces a full recompile of the
    500-step rollout scan (minutes per compile; a multi-round hardening sweep pays it
    a dozen times). Keeping params as an ARGUMENT lets one compiled evaluator serve
    every checkpoint -- see evaluator.make_param_evaluator.
    """
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    tgt = _resolve_target(target)
    env = tgt.load_env()
    pp = get_ppo_params(tgt.env_name, getattr(tgt, "domain", "locomotion"))
    network = _network_factory(pp)(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_inference = ppo_networks.make_inference_fn(network)

    def make_policy(params):
        return make_inference(params, deterministic=deterministic)

    return make_policy


def make_policy_fn(out_dir=None, params=None, target=None, deterministic=True):
    """Rebuild the deterministic inference fn for saved params.

    Reconstructs the same network (with observation normalization) used during
    training so that reloaded params produce identical distributions. Uses the
    target's env + checkpoint_name (default: go1).
    """
    tgt = _resolve_target(target)
    if params is None:
        params = load_params(out_dir, checkpoint_name=tgt.checkpoint_name)

    env = tgt.load_env()
    pp = get_ppo_params(tgt.env_name, getattr(tgt, "domain", "locomotion"))

    network = _network_factory(pp)(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )

    make_inference = ppo_networks.make_inference_fn(network)
    return make_inference(params, deterministic=deterministic)
