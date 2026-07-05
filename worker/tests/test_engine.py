import sys
import types

from app import engine


def test_load_engine_passes_det_tuning_params(monkeypatch):
    """Detection knobs (eval-tuned, env-overridable) must reach PaddleOCR init."""
    captured = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)
    # Pretend the ch detector is already downloaded so bootstrap is a no-op.
    monkeypatch.setattr(engine.os.path, "isdir", lambda path: True)

    engine._load_engine.cache_clear()
    try:
        engine._load_engine()
    finally:
        engine._load_engine.cache_clear()

    assert captured["det_limit_side_len"] == engine._DET_LIMIT_SIDE_LEN
    assert captured["det_db_unclip_ratio"] == engine._DET_DB_UNCLIP_RATIO
    assert captured["det_db_box_thresh"] == engine._DET_DB_BOX_THRESH
    assert captured["lang"] == "korean"
    assert captured["use_angle_cls"] is False
