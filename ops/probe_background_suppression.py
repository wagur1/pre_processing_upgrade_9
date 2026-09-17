#!/usr/bin/env python
"""Probe: detector-mask-driven background suppression, no training.

Why this exists. Every *learned* preprocessor the project tried on images failed:
the AR checkpoint costs ~12% more bits at equal detection mAP, and the
detection-trained one destroys 25% of the detector's mAP before the codec is even
involved (mAP(pre)/mAP(x) = 0.75). The detection literature's large numbers come
from the opposite philosophy — REMOVE background, keep objects (ROI-Packing
-44%, dual-region JPEG -26%, Rozek VCIP 2023) — which needs no learned editor and
therefore cannot overfit the proxy codec or add structure the detector dislikes.

Design: run the frozen detector on the SOURCE image (the encoder has the image
and may analyse it freely; the decoder needs no side information), dilate its
boxes into a protection mask, and heavily blur everything OUTSIDE the mask. The
object regions are passed through untouched. That is the whole transform — no
parameters, no training, nothing to select.

Arms: anchor (codec(x)) vs masked (codec(suppress(x))) on the mAP axis, with
several blur strengths, both codecs, the project's QP grid. Per-image records are
saved so the CI is recomputed offline (CPU) rather than in the kernel.

Usage:
  python ops/probe_background_suppression.py --images <val2017> --ann <instances_val2017.json> \
      --n-images 500 --size 320 --sigmas 4,8,16 --out outputs/probe_bgsuppress
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codecs.standard import StandardCodec, ffmpeg_available  # noqa: E402
from src.metrics.bd_rate import bd_rate  # noqa: E402
from probe_detection import (  # noqa: E402  (ops/ is on sys.path when run from repo root)
    Detector,
    _coco_box,
    coco_map,
    load_coco,
    scaled_gt,
)


def protect_mask(boxes, scores, labels, size: int, score_thresh: float,
                 dilate: float) -> torch.Tensor:
    """[1,1,S,S] mask, 1 = protect. Boxes are xyxy in the (already resized) frame."""
    m = torch.zeros(1, 1, size, size)
    keep = scores >= score_thresh
    for b in boxes[keep]:
        x1, y1, x2, y2 = b.tolist()
        w, h = x2 - x1, y2 - y1
        x1 -= dilate * w
        x2 += dilate * w
        y1 -= dilate * h
        y2 += dilate * h
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(size, int(np.ceil(x2))), min(size, int(np.ceil(y2)))
        if x2 > x1 and y2 > y1:
            m[:, :, y1:y2, x1:x2] = 1.0
    return m


def suppress(x: torch.Tensor, mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Heavy blur outside the mask; object regions byte-identical to the source.

    Separable Gaussian (depthwise, so RGB is blurred channel-wise) applied only
    where the mask is 0. Same 'destroy what the machine does not look at' idea as
    the AR design's M1, but driven by the consumer's own detections instead of a
    learned saliency head."""
    if sigma <= 0:
        return x
    k = int(2 * round(2 * sigma) + 1)
    c = x.shape[1]
    ax = torch.arange(k, dtype=x.dtype, device=x.device) - (k - 1) / 2
    g = torch.exp(-ax.pow(2) / (2 * sigma * sigma))
    g = (g / g.sum())
    gx = g.view(1, 1, k, 1).expand(c, 1, k, 1).contiguous()
    gy = g.view(1, 1, 1, k).expand(c, 1, 1, k).contiguous()
    b, _, t, h, w = x.shape
    flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    pad = k // 2
    blur = F.conv2d(F.pad(flat, (pad, pad, 0, 0), mode="reflect"), gx, groups=c)
    blur = F.conv2d(F.pad(blur, (0, 0, pad, pad), mode="reflect"), gy, groups=c)
    blur = blur.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    m = mask.expand_as(x)
    return x * m + blur * (1 - m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--qps", default="30,35,40,45,50")
    ap.add_argument("--sigmas", default="4,8,16")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--dilate", type=float, default=0.15)
    ap.add_argument("--out", default="outputs/probe_bgsuppress")
    a = ap.parse_args()

    if not ffmpeg_available():
        print("[bg] ffmpeg missing -> abort")
        raise SystemExit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigmas = [float(s) for s in a.sigmas.split(",")]
    qps = [int(q) for q in a.qps.split(",")]

    ann_meta, items = load_coco(Path(a.images), Path(a.ann), a.n_images, a.size, 0)
    items = [(i, t.to(device), hw, an) for i, t, hw, an in items]
    gt_by_id = {i: scaled_gt(an, a.size, hw) for i, _, hw, an in items}
    ids = [i for i, _, _, _ in items]
    print(f"[bg] {len(items)} images at {a.size}px, sigmas={sigmas}")

    det = Detector(device)

    def mAP(pred_list):
        return coco_map(pred_list, gt_by_id, ids, ann_meta)[0]

    from torchvision.ops import nms  # noqa: E402  (only needed for the mask tally)
    masks = {}
    for i, t, hw, _ in items:
        d = det.predict(t)[0]
        masks[i] = protect_mask(d["boxes"], d["scores"], d["labels"], a.size,
                                a.score, a.dilate)
    cover = float(np.mean([m.mean().item() for m in masks.values()]))
    print(f"[bg] mean protected fraction = {cover:.3f} "
          f"(1 - this is how much of the image may be destroyed)")

    arms = ["anchor"] + [f"blur{s:g}" for s in sigmas]
    rec = {arm: {} for arm in arms}
    for codec_name in ("h264", "h265"):
        for qp in qps:
            sc = StandardCodec(codec=codec_name, qp=qp, preset="medium")
            for arm in arms:
                rec[arm].setdefault((codec_name, qp), {})
            for i, t, hw, _ in items:
                variants = {"anchor": t}
                for s in sigmas:
                    variants[f"blur{s:g}"] = suppress(t, masks[i], s)
                for arm, xv in variants.items():
                    out, bpp = sc.compress_decompress_items(xv)
                    d = det.predict(out)[0]
                    keep = d["scores"] >= det.score_thresh
                    rec[arm][(codec_name, qp)][i] = (float(bpp[0]), [
                        {"image_id": i, "category_id": int(l), "bbox": _coco_box(b),
                         "score": float(s2)}
                        for b, s2, l in zip(d["boxes"][keep], d["scores"][keep],
                                            d["labels"][keep])])
            print(f"[bg] {codec_name} qp{qp} done", flush=True)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"n_images": len(items), "size": a.size, "cover": cover,
              "sigmas": sigmas, "score": a.score, "dilate": a.dilate, "curves": {}}
    for codec_name in ("h264", "h265"):
        curves = {}
        for arm in arms:
            rates, aps = [], []
            for qp in qps:
                slot = rec[arm][(codec_name, qp)]
                rates.append(float(np.mean([v[0] for v in slot.values()])))
                aps.append(mAP([p for v in slot.values() for p in v[1]]))
            curves[arm] = {"rate": rates, "mAP": aps}
            print(f"[bg] {codec_name} {arm:8s} bpp={['%.4f' % r for r in rates]} "
                  f"mAP={['%.4f' % m for m in aps]}")
        for arm in arms[1:]:
            curves[arm]["bd_vs_anchor"] = bd_rate(
                curves["anchor"]["rate"], curves["anchor"]["mAP"],
                curves[arm]["rate"], curves[arm]["mAP"])
            print(f"[bg] {codec_name} {arm}: BD = "
                  f"{curves[arm]['bd_vs_anchor']:+.2f}%")
        result["curves"][codec_name] = curves

    # records for the offline bootstrap CI
    flat = {}
    for arm in arms:
        for (codec_name, qp), slot in rec[arm].items():
            img = sorted(slot)
            boxes, offs, labs = [], [0], []
            for i in img:
                for p in slot[i][1]:
                    boxes.append(p["bbox"] + [p["score"]])
                    labs.append(p["category_id"])
                offs.append(len(boxes))
            tag = f"{arm}_{codec_name}_{qp}"
            flat[f"{tag}_img"] = np.asarray(img, dtype=np.int64)
            flat[f"{tag}_bpp"] = np.asarray([slot[i][0] for i in img], dtype=np.float32)
            flat[f"{tag}_boxes"] = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
            flat[f"{tag}_labels"] = np.asarray(labs, dtype=np.int32)
            flat[f"{tag}_offsets"] = np.asarray(offs, dtype=np.int64)
    np.savez_compressed(out_dir / "per_image_records.npz", **flat)

    def finite(o):
        if isinstance(o, dict):
            return {k: finite(v) for k, v in o.items()}
        if isinstance(o, list):
            return [finite(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    (out_dir / "probe_bgsuppress.json").write_text(json.dumps(finite(result), indent=2))
    print(f"[bg] wrote {out_dir / 'probe_bgsuppress.json'}")


if __name__ == "__main__":
    main()
