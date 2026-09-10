"""Assemble the v9 warm-start checkpoint: PRE from v7-v1 (best pre-only) +
trunk/POST from the v8 STE checkpoint, into a PerCodecPostSandwich state.

Run: python ops/make_v9_ckpt.py \
        --v1 /tmp/v1_ckpt/.../preprocessor.pth \
        --ste /tmp/ste_out/.../preprocessor.pth \
        --out outputs/v9_warmstart.pth
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.percodec_sandwich import PerCodecPostSandwich  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True, help="v7 UP-VCM best checkpoint (PRE)")
    ap.add_argument("--ste", required=True, help="v8 STE sandwich checkpoint (POST trunk)")
    ap.add_argument("--out", default="outputs/v9_warmstart.pth")
    a = ap.parse_args()

    m = PerCodecPostSandwich()
    v1 = torch.load(a.v1, map_location="cpu")
    ste = torch.load(a.ste, map_location="cpu")

    # PRE <- v1
    m.load_pre_state(v1["model"] if "model" in v1 else v1)
    # trunk <- v8 STE (percodec's load_v8_sandwich copies matching keys)
    rep = m.load_v8_sandwich(ste["model"] if "model" in ste else ste)
    print("warm-start report:", rep)

    ck = {"model": m.state_dict(), "opt": None, "sched": None,
          "cfg": ste.get("cfg", {}), "epoch": 0, "global_step": 0,
          "best_val": None, "no_improve": 0}
    # stamp the arch so eval builds the right class
    ck["cfg"] = dict(ck["cfg"] or {})
    ck["cfg"]["model"] = dict(ck["cfg"].get("model") or {})
    ck["cfg"]["model"]["arch"] = "percodec_sandwich"
    ck["cfg"]["model"]["post_base"] = 32
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a.out)
    ms = ck["model"]
    print(f"saved {a.out}: PRE dec={ms['pre.dec_strength'].item():+.4f} "
          f"stab={ms['pre.stab_strength'].item():+.4f} "
          f"POST strength={ms['post_strength'].item():+.4f}")


if __name__ == "__main__":
    main()
