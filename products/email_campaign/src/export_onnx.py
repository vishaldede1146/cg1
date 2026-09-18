"""
Exports the trained multimodal fusion model (best_fusion_model.pt) to
ONNX, then applies post-training dynamic INT8 quantization.

Produces:
    models/fusion_model.onnx              (FP32 ONNX export)
    models/fusion_model_quantized.onnx     (dynamic INT8 quantized)

Notes on quantization scope:
    Dynamic quantization (onnxruntime.quantization.quantize_dynamic)
    quantizes weights of Linear/MatMul/Gemm ops to INT8 (activations are
    quantized on-the-fly at runtime). This gives strong size/speed wins
    on the DistilBERT encoder and the MLP/fusion/head layers, which are
    almost entirely Linear/MatMul. EfficientNet's Conv2d layers are
    largely untouched by dynamic quantization -- if you also want the
    image branch quantized, you'd need *static* quantization with a
    calibration dataset (see the STATIC_QUANTIZATION_NOTES section at
    the bottom of this file for how to extend this script).

Run:
    pip install onnx onnxruntime
    python src/export_onnx.py
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.evaluate import load_model
from transformers import AutoTokenizer

ONNX_FP32_PATH = os.path.join(config.MODELS_DIR, "fusion_model.onnx")
ONNX_INT8_PATH = os.path.join(config.MODELS_DIR, "fusion_model_quantized.onnx")

OPSET_VERSION = 17  # EfficientNet's SiLU/Swish activation needs opset >= 14


# ----------------------------------------------------------------------
# 1. Export wrapper: single tensor output instead of (tensor, dict)
# ----------------------------------------------------------------------
class ONNXExportWrapper(nn.Module):
    """torch.onnx.export needs a plain tensor (or tuple of tensors) output,
    not a dict, so this wrapper drops the dict branch and only returns
    the stacked [batch, n_targets] prediction tensor, in the fixed
    order given by config.TARGET_COLUMNS."""

    def __init__(self, fusion_model):
        super().__init__()
        self.fusion_model = fusion_model

    def forward(self, tabular, input_ids, attention_mask, image):
        stacked_pred, _ = self.fusion_model(tabular, input_ids, attention_mask, image)
        return stacked_pred


# ----------------------------------------------------------------------
# 2. Build dummy inputs matching the model's real input shapes/dtypes
# ----------------------------------------------------------------------
def build_dummy_inputs(tabular_dim, batch_size=1, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    encoded = tokenizer(
        ["20% off everything today [SEP] hurry, this deal wont last long"] * batch_size,
        padding="max_length", truncation=True,
        max_length=config.TEXT_MAX_LENGTH, return_tensors="pt",
    )

    tabular = torch.randn(batch_size, tabular_dim, dtype=torch.float32, device=device)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    image = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE,
                         dtype=torch.float32, device=device)
    return tabular, input_ids, attention_mask, image


# ----------------------------------------------------------------------
# 3. Export to ONNX (FP32)
# ----------------------------------------------------------------------
def export_to_onnx(model, tabular_dim, output_path=ONNX_FP32_PATH):
    model.eval()
    wrapper = ONNXExportWrapper(model).eval()

    tabular, input_ids, attention_mask, image = build_dummy_inputs(tabular_dim)

    dynamic_axes = {
        "tabular": {0: "batch_size"},
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "image": {0: "batch_size"},
        "predictions": {0: "batch_size"},
    }

    print(f"Exporting to ONNX (opset {OPSET_VERSION}) ...")
    torch.onnx.export(
        wrapper,
        (tabular, input_ids, attention_mask, image),
        output_path,
        input_names=["tabular", "input_ids", "attention_mask", "image"],
        output_names=["predictions"],
        dynamic_axes=dynamic_axes,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )
    print(f"Saved FP32 ONNX model to {output_path}")
    return tabular, input_ids, attention_mask, image


# ----------------------------------------------------------------------
# 4. Verify ONNX output matches PyTorch output
# ----------------------------------------------------------------------
def verify_onnx(model, onnx_path, dummy_inputs, atol=1e-3):
    import onnxruntime as ort

    tabular, input_ids, attention_mask, image = dummy_inputs

    with torch.no_grad():
        torch_out, _ = model(tabular, input_ids, attention_mask, image)
    torch_out = torch_out.numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(
        None,
        {
            "tabular": tabular.numpy(),
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "image": image.numpy(),
        },
    )[0]

    max_diff = np.max(np.abs(torch_out - onnx_out))
    print(f"Max abs diff PyTorch vs ONNX: {max_diff:.6f}  (tolerance={atol})")
    if max_diff > atol:
        print("WARNING: difference exceeds tolerance -- inspect before deploying.")
    else:
        print("ONNX export verified: outputs match PyTorch within tolerance.")
    return max_diff


# ----------------------------------------------------------------------
# 5. Dynamic INT8 quantization
# ----------------------------------------------------------------------
def quantize_onnx_model(fp32_path=ONNX_FP32_PATH, int8_path=ONNX_INT8_PATH):
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("Running dynamic INT8 quantization (weights of Linear/MatMul/Gemm ops) ...")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QUInt8,
    )
    print(f"Saved quantized model to {int8_path}")


# ----------------------------------------------------------------------
# 6. Size + latency comparison
# ----------------------------------------------------------------------
def benchmark(fp32_path, int8_path, dummy_inputs, n_runs=30):
    import onnxruntime as ort

    tabular, input_ids, attention_mask, image = dummy_inputs
    feed = {
        "tabular": tabular.numpy(),
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "image": image.numpy(),
    }

    def timed_run(path):
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        # warmup
        for _ in range(3):
            session.run(None, feed)
        start = time.time()
        for _ in range(n_runs):
            session.run(None, feed)
        elapsed = (time.time() - start) / n_runs
        return elapsed

    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    fp32_latency = timed_run(fp32_path)
    int8_latency = timed_run(int8_path)

    print("\n===== ONNX Export / Quantization Summary =====")
    print(f"{'Model':<12} {'Size (MB)':>12} {'Avg latency (ms)':>20}")
    print(f"{'FP32':<12} {fp32_size:>12.2f} {fp32_latency * 1000:>20.2f}")
    print(f"{'INT8':<12} {int8_size:>12.2f} {int8_latency * 1000:>20.2f}")
    print(f"Size reduction: {(1 - int8_size / fp32_size) * 100:.1f}%")
    print(f"Speedup: {fp32_latency / int8_latency:.2f}x  (CPU, batch_size={tabular.shape[0]})")
    print("================================================\n")


def main():
    print("Loading trained PyTorch model ...")
    model, checkpoint = load_model(device="cpu")  # ONNX export must be done on CPU
    tabular_dim = checkpoint["tabular_input_dim"]
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.5f})")

    dummy_inputs = export_to_onnx(model, tabular_dim)
    verify_onnx(model, ONNX_FP32_PATH, dummy_inputs)
    quantize_onnx_model()
    benchmark(ONNX_FP32_PATH, ONNX_INT8_PATH, dummy_inputs)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# STATIC_QUANTIZATION_NOTES
# ----------------------------------------------------------------------
# To also quantize the EfficientNet convolutional layers (dynamic
# quantization mostly skips Conv2d), use static quantization instead:
#
#   from onnxruntime.quantization import quantize_static, CalibrationDataReader
#
#   class CampaignCalibrationReader(CalibrationDataReader):
#       def __init__(self, calibration_batches):
#           self._iter = iter(calibration_batches)  # list of feed dicts
#       def get_next(self):
#           return next(self._iter, None)
#
#   quantize_static(
#       model_input=ONNX_FP32_PATH,
#       model_output="models/fusion_model_static_int8.onnx",
#       calibration_data_reader=CampaignCalibrationReader(calibration_batches),
#   )
#
# This needs ~100-500 representative real (tabular, input_ids,
# attention_mask, image) batches from your validation set as feed
# dicts, since static quantization calibrates activation ranges from
# real data rather than quantizing weights only. It usually gives a
# bigger speed/size win but takes more setup and a small extra
# accuracy check afterward (re-run evaluate-style metrics on the
# quantized model's outputs vs the FP32 ONNX outputs).
