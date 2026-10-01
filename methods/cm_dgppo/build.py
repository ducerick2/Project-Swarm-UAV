"""Dựng env có nhiễu và thuật toán từ config YAML (dùng chung cho train_cm.py và eval_cm.py)."""
from __future__ import annotations

import copy

from .noisy_env import make_noisy_env

# method (YAML) -> cm_variant của CMDGPPO; "dgppo" là DGPPO gốc
METHOD_TO_VARIANT = {
    "cm_dgppo_state": "state",
    "cm_dgppo_cbf": "cbf",
    "cm_dgppo": "full",
    "cm_dgppo_full": "full",
    "cm_dgppo_scalar": "scalar",
    "fixed_margin": "fixed",
}

# siêu tham số mặc định của DGPPO (third_party/dgppo/train.py)
DGPPO_DEFAULTS = dict(
    actor_gnn_layers=2, Vl_gnn_layers=2, Vh_gnn_layers=1, rnn_layers=1, lr_actor=3e-4, lr_Vl=1e-3,
    lr_Vh=1e-3, max_grad_norm=2.0, alpha=10.0, cbf_eps=1e-2, cbf_weight=1.0, batch_size=16384,
    use_rnn=True, use_lstm=False, coef_ent=1e-2, rnn_step=16, gamma=0.99, clip_eps=0.25,
    cbf_schedule=True,
)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """`--set a.b=giá_trị` (giá trị đọc bằng YAML). Trả bản sao đã sửa."""
    import yaml

    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def build_env(cfg: dict, sigma_w=None, sigma_v=None, n_obs=None, num_agents=None):
    noise = cfg.get("noise", {})
    cm = cfg.get("cm", {})
    return make_noisy_env(
        env_id=cfg["env"],
        num_agents=cfg["N"] if num_agents is None else num_agents,
        sigma_w=noise.get("sigma_w", 0.0) if sigma_w is None else sigma_w,
        sigma_v=noise.get("sigma_v", 0.0) if sigma_v is None else sigma_v,
        num_obs=cfg.get("n_obs", 3) if n_obs is None else n_obs,
        sigma_base=cm.get("sigma_base", 0.05),
        sigma_kappa=cm.get("sigma_kappa", 5.0),
    )


def build_algo(cfg: dict, env, seed: int, steps: int, batch_size: int, policy_only: bool = False):
    """policy_only=True: luôn dựng DGPPO (cùng kiến trúc actor) — đủ để eval/triển khai."""
    from dgppo.algo import DGPPO

    from .algo import CMDGPPO

    kwargs = DGPPO_DEFAULTS | dict(
        env=env, node_dim=env.node_dim, edge_dim=env.edge_dim, state_dim=env.state_dim,
        action_dim=env.action_dim, n_agents=env.num_agents, seed=seed, train_steps=steps,
        batch_size=batch_size,
    )
    method = cfg["method"]
    if method == "dgppo" or policy_only:
        return DGPPO(**kwargs)
    if method in METHOD_TO_VARIANT:
        return CMDGPPO(**kwargs, cm_variant=METHOD_TO_VARIANT[method], **cfg.get("cm", {}))
    raise ValueError(f"không hỗ trợ method={method}")
