import pytest

torch = pytest.importorskip("torch")


def test_export_keeps_only_ema(tmp_path):
    from train.export_ema import export_ema
    src, dst = tmp_path / "model_10.pt", tmp_path / "ema.pt"
    torch.save({"model_state_dict": {"w": torch.ones(2)}, "optimizer_state_dict": {"s": 1},
                "ema_model_state_dict": {"ema_model.w": torch.zeros(2)}, "step": 10}, src)
    export_ema(src, dst)
    out = torch.load(dst, weights_only=True)
    assert set(out) == {"ema_model_state_dict"}
    assert torch.equal(out["ema_model_state_dict"]["ema_model.w"], torch.zeros(2))


def test_warm_state_strips_ema_prefix_and_counters():
    from train.finetune import warm_state
    ck = {"ema_model_state_dict": {"initted": torch.tensor(True), "step": torch.tensor(5),
                                   "ema_model.transformer.x": torch.ones(1)}, "step": 5}
    assert set(warm_state(ck)) == {"transformer.x"}
