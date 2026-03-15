"""Configuration loading for DNBN system and model configs."""

import json
import os


def load_sys_config(sys_config_path, repo_root=None):
    """Load a system DNBN config and resolve referenced DNBN model configs.

    Args:
        sys_config_path: Path to the sys_dnbn_*.json file.
        repo_root: Repository root for resolving relative config paths.

    Returns:
        dict with the full system configuration, each node enriched
        with a 'params' key containing the loaded DNBN parameters.
    """
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(sys_config_path, 'r') as f:
        sys_config = json.load(f)

    for node_id, node_cfg in sys_config['nodes'].items():
        dnbn_path = node_cfg['config']
        if not os.path.isabs(dnbn_path):
            dnbn_path = os.path.join(repo_root, dnbn_path)
        with open(dnbn_path, 'r') as f:
            node_cfg['params'] = json.load(f)

    return sys_config
