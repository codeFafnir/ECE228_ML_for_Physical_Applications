import sys, os, traceback

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PINN_channel-estimation-main'))

try:
    print("Step 1: importing shared...")
    from shared.model_loader import load_fp32_model
    from shared.calibration_data import get_synthetic_loaders
    from shared.eval_utils import evaluate
    print("Step 2: shared OK")

    import torch, copy, time

    print("Step 3: building model...")
    model = load_fp32_model(None, rss_size=30)
    print("Step 4: model OK")

    cal_loader, val_loader = get_synthetic_loaders(n_cal=8, n_val=8, batch_size=4)
    print("Step 5: loaders OK")

    device = torch.device('cpu')
    r = evaluate(model, val_loader, device)
    print(f"Step 6: FP32 NMSE = {r['nmse_db']:.3f} dB")

    print("Step 7: importing GPTQ...")
    from method_gptq.gptq_quantize_pinn import gptq_quantize_pinn
    print("Step 8: GPTQ import OK")

    t0 = time.perf_counter()
    model_q = gptq_quantize_pinn(
        model=copy.deepcopy(model),
        cal_loader=cal_loader,
        num_bits=3,
        group_size=64,
        device=device,
        verbose=True,
    )
    print(f"Step 9: GPTQ done in {time.perf_counter()-t0:.1f}s")

    r_q = evaluate(model_q, val_loader, device)
    print(f"Step 10: GPTQ NMSE = {r_q['nmse_db']:.3f} dB  (delta: {r_q['nmse_db']-r['nmse_db']:+.3f} dB)")

except Exception:
    traceback.print_exc()
    sys.exit(1)

print("ALL OK")
