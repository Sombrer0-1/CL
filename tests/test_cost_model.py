from orion_repro.memory.cost_model import MemoryCostModel, fit_batch_slope


def test_fit_batch_slope_exact_line():
    intercept, slope, r2 = fit_batch_slope([8, 16, 32], [1008, 1016, 1032])
    assert abs(slope - 1.0) < 1e-6
    assert abs(intercept - 1000.0) < 1e-6
    assert r2 > 0.999


def test_device_predict_uses_effective_batch_not_host_frames():
    model = MemoryCostModel(
        intercept_bytes=1000,
        m_batch_bytes=10,
        m_frame_bytes=999999,
        plugin_gem_ewc_bytes=50,
        r2=1.0,
        m_frame_raw_uint8_bytes=3072,
        m_frame_latent_f32_bytes=32776,
    )
    device = model.predict_bytes(16, 200, replay_batch=16, advanced=False, resource="device")
    assert device == 1000 + 10 * (16 + 16)
    adv = model.predict_bytes(16, 200, replay_batch=16, advanced=True, resource="device")
    assert adv - device == 50
    host_raw = model.predict_bytes(16, 200, resource="host", representation="raw")
    assert host_raw == 3072 * 200
    host_lat = model.predict_host_replay_bytes(10, representation="latent")
    assert host_lat == 32776 * 10
