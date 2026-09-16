#!/usr/bin/env python
"""Zero-shot detection probe: does the VCM preprocessor transfer to object detection?

Standalone (no engine changes). Two stages in one run, because stage A is cheap
and can kill the direction before stage B burns quota:

  Stage A (no codec): mAP of a frozen COCO detector on x vs pre(x).
      The PRE was trained on Kinetics clips at 128px. If its edit already
      destroys detection accuracy at the probe resolution, no rate-accuracy
      sweep can rescue it -> stop and report.

  Stage B (codec sweep): the canonical 3-arm protocol on the mAP axis --
      anchor   codec(x)                 -> detector
      prep     codec(pre(x))            -> detector
      sandwich codec(pre(x)) -> post    -> detector
      BD-rate(mAP) per codec with a bootstrap CI over images.

Everything else mirrors the group's protocol: real x264/x265 preset medium,
QP 30-50, frozen checkpoints, per-image records so the CI is computed the same
way as ops/merge_eval.py.

Usage (Kaggle):
  python ops/probe_detection.py --images /kaggle/input/coco-2017-dataset/coco2017 \\
      --ckpt <preprocessor.pth> --config configs/sandwich_ar.yaml \\
      --n-images 500 --size 320 --stage both --out outputs/probe_detection
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codecs.standard import StandardCodec, ffmpeg_available  # noqa: E402
from src.config import apply_overrides, load_config  # noqa: E402
from src.engine import _build_models, _qp_norm, _rate_cond  # noqa: E402
from src.metrics.bd_rate import bd_rate  # noqa: E402


# ---------------------------------------------------------------- data -----
def load_coco(images_dir: Path, ann_file: Path, n: int, size: int, seed: int = 0):
    """Return [(image_id, tensor[1,3,1,H,W], orig_hw)] for n val images."""
    from PIL import Image

    ann = json.loads(Path(ann_file).read_text())
    id_to_ann = {}
    for a in ann["annotations"]:
        id_to_ann.setdefault(a["image_id"], []).append(a)
    imgs = [im for im in ann["images"] if im["id"] in id_to_ann]
    rng = random.Random(seed)
    rng.shuffle(imgs)
    out = []
    for im in imgs[:n]:
        p = Path(images_dir) / im["file_name"]
        if not p.is_file():
            continue
        img = Image.open(p).convert("RGB")
        w0, h0 = img.size
        img = img.resize((size, size), Image.BILINEAR)
        arr = torch.from_numpy(np.asarray(img)).float().div_(255.0)     # [H,W,3]
        t = arr.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)              # [1,3,1,H,W]
        out.append((im["id"], t, (h0, w0), id_to_ann[im["id"]]))
    return ann, out


def scaled_gt(anns, size: int, orig_hw):
    """COCO gt boxes rescaled from the original image to the probe resolution."""
    h0, w0 = orig_hw
    out = []
    for a in anns:
        x, y, w, h = a["bbox"]
        out.append({
            "id": a["id"], "image_id": a["image_id"],
            "category_id": a["category_id"], "iscrowd": a.get("iscrowd", 0),
            "area": (w * size / w0) * (h * size / h0),
            "bbox": [x * size / w0, y * size / h0, w * size / w0, h * size / h0],
        })
    return out


# ------------------------------------------------------------- detector ----
class Detector:
    """Frozen COCO Faster R-CNN. CTC uses Detectron2 X101-FPN; this is the
    drop-in stand-in available without a Detectron2 install (torchvision)."""

    def __init__(self, device, score_thresh: float = 0.05):
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn)
        self.device = device
        self.score_thresh = score_thresh
        self.model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1).to(device).eval()

    @torch.no_grad()
    def predict(self, imgs: torch.Tensor):
        """imgs [B,3,H,W] or [B,3,T,H,W] -> list of dicts (boxes/scores/labels).

        The probe feeds single-frame items, so a 5-D clip tensor is squeezed to
        its first frame; torchvision wants a list of [3,H,W]."""
        x = imgs.to(self.device)
        if x.ndim == 5:
            x = x[:, :, 0]
        return self.model(list(x))


def coco_map(results, gt_by_id, image_ids, ann_meta):
    """mAP@[.5:.95] and mAP@0.5 for the given predictions (pycocotools)."""
    from pycocotools import mask as _mask  # noqa: F401
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = {"images": [{"id": i} for i in image_ids],
                       "annotations": [a for i in image_ids for a in gt_by_id[i]],
                       "categories": ann_meta["categories"]}
    coco_gt.createIndex()
    if not results:
        return 0.0, 0.0
    dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, dt, "bbox")
    ev.params.imgIds = list(image_ids)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])


# ---------------------------------------------------------------- probe ----
def run(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = Path(args.images)
    ann_file = Path(args.ann) if args.ann else images_dir.parent / "annotations" / "instances_val2017.json"
    print(f"[probe] images={images_dir} ann={ann_file} device={device}")

    ann_meta, items = load_coco(images_dir, ann_file, args.n_images, args.size, args.seed)
    print(f"[probe] {len(items)} images at {args.size}px")
    gt_by_id = {i: scaled_gt(a, args.size, hw) for i, _, hw, a in items}
    image_ids = [i for i, _, _, _ in items]

    cfg = apply_overrides(load_config(args.config),
                          [f"device={device.type}", "model.post_base=64"])
    pre, codec, _ = _build_models(cfg, device, role="eval")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pre.load_state_dict(state["model"] if "model" in state else state)
    pre.eval()
    print(f"[probe] PRE+POST loaded from {args.ckpt}")

    det = Detector(device)
    results: dict = {"n_images": len(items), "size": args.size, "stages": {}}

    # ---------------- stage A: no codec, does the edit preserve detection? --
    def mAP_of(fn, tag, subset=None):
        ids = image_ids if subset is None else subset
        preds = []
        for i, t, hw, _ in items:
            if i not in ids:
                continue
            d = fn(t)[0]
            keep = d["scores"] >= det.score_thresh
            for b, s, l in zip(d["boxes"][keep], d["scores"][keep], d["labels"][keep]):
                preds.append({"image_id": i, "category_id": int(l),
                              "bbox": [float(v) for v in b.tolist()], "score": float(s)})
        coco_ap, ap50 = coco_map(preds, gt_by_id, ids, ann_meta)
        print(f"[probe] {tag}: mAP={coco_ap:.4f} mAP@.5={ap50:.4f} "
              f"({len(preds)} boxes >= {det.score_thresh} over {len(ids)} images)")
        return coco_ap, len(preds)

    def det_plain(t):
        return det.predict(t)

    def det_pre(t):
        with torch.no_grad():
            xp = pre(t, _rate_cond(_qp_norm(40, cfg), 1, t.device, t.dtype))
        return det.predict(xp)

    ap_x, n_boxes = mAP_of(det_plain, "stageA anchor (x)")
    ap_p, _ = mAP_of(det_pre, "stageA prep (pre(x))")
    results["stages"]["A"] = {"mAP_x": ap_x, "mAP_pre": ap_p,
                              "boxes_anchor": n_boxes,
                              "ratio": (ap_p / ap_x) if ap_x > 0 else None}
    if n_boxes < args.min_anchor_boxes:
        print(f"[probe] SETUP PROBLEM: the anchor produced only {n_boxes} boxes "
              f"(< {args.min_anchor_boxes}) — the detector/resolution pairing is "
              f"wrong (not a PRE effect). Aborting so quota is not spent on a "
              f"degenerate anchor curve.")
        results["verdict"] = "degenerate_anchor"
        return results
    if ap_x > 0 and ap_p / ap_x < args.stage_a_threshold:
        print(f"[probe] STAGE A FAILED: pre(x) keeps only {ap_p / ap_x:.2f} of the "
              f"anchor mAP (< {args.stage_a_threshold}) -> the edit destroys "
              f"detection content at this resolution; stopping.")
        results["verdict"] = "stage_A_fail"
        return results

    if args.stage == "a":
        results["verdict"] = "stage_A_only"
        return results

    # ---------------- stage B: rate-accuracy sweep on the mAP axis ----------
    if not ffmpeg_available():
        print("[probe] ffmpeg missing -> stage B skipped")
        results["verdict"] = "no_ffmpeg"
        return results

    qps = [int(q) for q in args.qps.split(",")]
    arms = ["anchor", "prep", "sandwich"]
    per_image = {arm: {} for arm in arms}       # arm -> qp -> {image_id: (bpp, preds)}
    for codec_name in ("h264", "h265"):
        for qp in qps:
            sc = StandardCodec(codec=codec_name, qp=qp, preset="medium")
            for arm in arms:
                per_image[arm].setdefault((codec_name, qp), {})
            for i, t, hw, _ in items:
                with torch.no_grad():
                    xp = pre(t, _rate_cond(_qp_norm(qp, cfg), 1, t.device, t.dtype))
                rec_a, bpp_a = sc.compress_decompress_items(t)
                rec_p, bpp_p = sc.compress_decompress_items(xp)
                rec_s = pre.post_restore(rec_p, _rate_cond(_qp_norm(qp, cfg), 1,
                                                           rec_p.device, rec_p.dtype))
                for arm, rec, bpp in (("anchor", rec_a, bpp_a), ("prep", rec_p, bpp_p),
                                      ("sandwich", rec_s, bpp_p)):
                    d = det.predict(rec[:, :, 0])[0]
                    keep = d["scores"] >= det.score_thresh
                    preds = [{"image_id": i, "category_id": int(l),
                              "bbox": [float(v) for v in b.tolist()], "score": float(s)}
                             for b, s, l in zip(d["boxes"][keep], d["scores"][keep],
                                                d["labels"][keep])]
                    per_image[arm][(codec_name, qp)][i] = (float(bpp[0]), preds)
            print(f"[probe] coded {codec_name} qp{qp} for all arms")

    results["stages"]["B"] = {}
    for codec_name in ("h264", "h265"):
        curves = {}
        for arm in arms:
            rates, aps = [], []
            preds = [p for qp in qps for i, _ in [(k, v) for k, v in
                     per_image[arm][(codec_name, qp)].items()] for p in _[1]]
            for qp in qps:
                slot = per_image[arm][(codec_name, qp)]
                rates.append(float(np.mean([v[0] for v in slot.values()])))
                preds = [p for v in slot.values() for p in v[1]]
                aps.append(coco_map(preds, gt_by_id, image_ids, ann_meta)[0])
            curves[arm] = {"rate": rates, "mAP": aps}
            print(f"[probe] {codec_name} {arm}: bpp={['%.4f' % r for r in rates]} "
                  f"mAP={['%.4f' % a for a in aps]}")
        entry = {"curves": curves}
        for arm in ("prep", "sandwich"):
            entry[f"bd_{arm}"] = bd_rate(curves["anchor"]["rate"], curves["anchor"]["mAP"],
                                         curves[arm]["rate"], curves[arm]["mAP"])
        # bootstrap over images (resample the image set, recompute mAP per point)
        if args.bootstrap:
            draws = []
            rng = random.Random(0)
            for _ in range(args.bootstrap):
                sub = [i for i in image_ids if rng.random() < 0.8]
                cur = {}
                for arm in arms:
                    rates, aps = [], []
                    for qp in qps:
                        slot = per_image[arm][(codec_name, qp)]
                        rates.append(float(np.mean([slot[i][0] for i in sub])))
                        preds = [p for i in sub for p in slot[i][1]]
                        aps.append(coco_map(preds, gt_by_id, sub, ann_meta)[0])
                    cur[arm] = {"rate": rates, "mAP": aps}
                for arm in ("prep", "sandwich"):
                    draws.append(bd_rate(cur["anchor"]["rate"], cur["anchor"]["mAP"],
                                         cur[arm]["rate"], cur[arm]["mAP"]))
            ok = [d for d in draws if np.isfinite(d)]
            if ok:
                entry["ci"] = {"lo": float(np.percentile(ok, 2.5)),
                               "hi": float(np.percentile(ok, 97.5)),
                               "n_draws": len(ok)}
        results["stages"]["B"][codec_name] = entry
    results["verdict"] = "ran"
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/sandwich_ar.yaml")
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--qps", default="30,35,40,45,50")
    ap.add_argument("--stage", choices=["a", "both"], default="both")
    ap.add_argument("--stage-a-threshold", type=float, default=0.85,
                    help="abort if pre(x) mAP / anchor mAP falls below this")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--min-anchor-boxes", type=int, default=1,
                    help="abort if the anchor yields fewer detections: a broken "
                         "detector/resolution setup, not a PRE effect")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/probe_detection")
    a = ap.parse_args()
    res = run(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def _finite(o):
        """NaN/inf are not valid JSON and break downstream merges."""
        if isinstance(o, dict):
            return {k: _finite(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_finite(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    (out / "probe_detection.json").write_text(json.dumps(_finite(res), indent=2))
    print(f"\n[probe] verdict={res.get('verdict')}")
    print(json.dumps(_finite(res.get("stages", {}).get("B", {})), indent=2)[:1500])
    print(f"[probe] wrote {out / 'probe_detection.json'}")


if __name__ == "__main__":
    main()
