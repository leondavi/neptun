"""Create DNBN node and system configs for the YOLO comparison sweep."""

import json
from pathlib import Path


def main():
    # Create small DNBN node configs
    for mc in [32, 24, 16]:
        cfg = {
            "M": mc,
            "C": mc,
            "num_heads": 4,
            "buffer_size": 8,
            "controller_hidden": 64,
            "dropout": 0.1,
        }
        path = Path(f"configs/dnbn_m{mc}_c{mc}.json")
        if not path.exists():
            path.write_text(json.dumps(cfg, indent=4) + "\n")
            print(f"Created {path}")
        else:
            print(f"Already exists: {path}")

    # Create 8-node system configs for each
    base = json.loads(Path("configs/sys_dnbn_cifar10_8node_tuned.json").read_text())
    for mc in [32, 24, 16]:
        out_name = f"sys_dnbn_cifar10_8node_m{mc}c{mc}_tuned.json"
        out_path = Path("configs") / out_name
        if out_path.exists():
            print(f"Already exists: {out_path}")
            continue
        cfg = json.loads(json.dumps(base))
        cfg["name"] = f"CIFAR-10 8-Node DNBN (Tuned, M{mc} C{mc})"
        for node in cfg["nodes"].values():
            node["config"] = f"configs/dnbn_m{mc}_c{mc}.json"
        out_path.write_text(json.dumps(cfg, indent=4) + "\n")
        print(f"Created {out_path}")


if __name__ == "__main__":
    main()
