"""Tests for the v9 per-codec POST model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.percodec_sandwich import PerCodecPostSandwich  # noqa: E402
from src.models.sandwich import SandwichPreprocessor  # noqa: E402
from src.models.upvcm import UPVCMPreprocessor  # noqa: E402


def _clip(b=2, t=8, s=64):
    return torch.rand(b, 3, t, s, s)


def test_identity_at_init():
    torch.manual_seed(0)
    m = PerCodecPostSandwich()
    x = _clip()
    cond = torch.full((2, 1), 0.6)
    assert torch.allclose(m(x, cond), x, atol=1e-6)
    x_hat = torch.rand_like(x)
    for codec in ("h264", "h265"):
        assert torch.allclose(m.post_restore(x_hat, cond, codec=codec), x_hat,
                              atol=1e-6)


def test_codec_routing_changes_output():
    """The whole point of v9: different codecs must give different restores
    once the routing is active. At INIT the codec FiLM is zero (warm-start
    exactness), so this test un-zeros the FiLM final layer first — that is
    the state after a few STE steps of training."""
    m = PerCodecPostSandwich()
    x_hat = _clip()
    cond = torch.full((2, 1), 0.6)
    with torch.no_grad():
        m.post_net.out_conv.weight.normal_(0, 0.05)
        m.post_net.codec_embed.weight.normal_(0, 0.5)   # distinct embeddings
        # activate the codec path: FiLM final layer non-zero
        m.post_net.film.net[2].weight.normal_(0, 0.05)
        m.post_net.film.net[2].bias.normal_(0, 0.05)
        m.post_strength.fill_(0.5)
        r264 = m.post_restore(x_hat, cond, codec="h264")
        r265 = m.post_restore(x_hat, cond, codec="h265")
    assert not torch.allclose(r264, r265, atol=1e-6), "codec routing is a no-op"


def test_gate_gradients_alive_at_init():
    """Dead-saddle regression guard (v7/v8 lesson)."""
    torch.manual_seed(0)
    m = PerCodecPostSandwich()
    x_hat = _clip()
    out = m.post_restore(x_hat, torch.full((2, 1), 0.6), codec="h264")
    out.pow(2).mean().backward()
    assert m.post_strength.grad is not None and m.post_strength.grad.abs() > 0
    assert m.post_net.codec_embed.weight.grad is not None


def test_v8_warmstart_exact_at_load():
    """v8 checkpoint loaded via load_v8_sandwich: codec-agnostic output must
    EQUAL v8's (zero-init codec FiLM contributes nothing)."""
    torch.manual_seed(0)
    v8 = SandwichPreprocessor()
    with torch.no_grad():  # make gates nonzero so identity isn't trivial
        v8.post_net.out_conv.weight.normal_(0, 0.05)
        v8.post_strength.fill_(0.3)
        v8.pre.dec_strength.fill_(-0.4)
    v9 = PerCodecPostSandwich()
    rep = v9.load_v8_sandwich(v8.state_dict())
    # 56 v8 keys total: 4 film keys skipped (different cond width),
    # 1 shape-mismatched → ~51-52 copied. Everything copyable is copied.
    assert rep["copied"] >= 50, rep
    assert rep["missing"] == [], rep["missing"]
    x_hat = _clip(b=1, t=4, s=32)
    cond = torch.full((1, 1), 0.5)
    with torch.no_grad():
        o8 = v8.post_restore(x_hat, cond)
        o9 = v9.post_restore(x_hat, cond, codec="h264")
        o9b = v9.post_restore(x_hat, cond, codec="h265")
    assert torch.allclose(o8, o9, atol=1e-5), "warm-start changed h264 behavior"
    assert torch.allclose(o8, o9b, atol=1e-5), "warm-start changed h265 behavior"


def test_engine_compat_signature():
    """post_restore must accept the engine's keyword call."""
    m = PerCodecPostSandwich()
    x_hat = _clip(b=1, t=4, s=32)
    out = m.post_restore(x_hat, torch.zeros(1, 1), codec="h265")
    assert out.shape == x_hat.shape


def test_param_count():
    m = PerCodecPostSandwich()
    n = sum(p.numel() for p in m.parameters())
    # ~44k PRE + ~577k trunk + codec embed/FiLM delta (~5k)
    assert 600_000 < n < 700_000, f"unexpected param count {n}"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all v9 tests passed")
