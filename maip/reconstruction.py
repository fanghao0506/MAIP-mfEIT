"""Instance optimization of the authors' MBA-Net against voltage data."""
from dataclasses import asdict, dataclass
import time

import numpy as np
import torch

from .data import normalize_sensitivity
from .model import MultiBranchNetwork


@dataclass(frozen=True)
class Config:
    epochs: int = 950
    learning_rate: float = 1.2e-4
    seed: int = 2


def select_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in ("cpu", "cuda"):
        raise ValueError("Device must be auto, cpu or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    return torch.device(name)


def reconstruct(measurement, config=Config(), device="auto", progress=True):
    if config.epochs < 1 or config.learning_rate <= 0:
        raise ValueError("Epoch count and learning rate must be positive")
    if measurement.voltages.shape != (measurement.sensitivity.shape[0], 4) or not np.isfinite(measurement.voltages).all():
        raise ValueError("Expected finite voltages with one row per measurement and four frequency columns")
    device = select_device(device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    preparation_started = time.perf_counter()
    eit2rect = torch.from_numpy(measurement.eit2rect).to(device)
    rect2eit = torch.from_numpy(measurement.rect2eit).to(device)
    model = MultiBranchNetwork().to(device)
    voltages = torch.from_numpy(measurement.voltages).to(device)
    operator = torch.from_numpy(normalize_sensitivity(measurement.sensitivity)).to(device).matmul(rect2eit)
    # Draw on CPU before transfer, as in the original inference script.
    noise = torch.rand(1, 4, 64, 64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.L1Loss()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - preparation_started
    model.train()
    losses = []
    started = time.perf_counter()
    for iteration in range(config.epochs):
        optimizer.zero_grad()
        images = model(noise).squeeze(0)
        coefficients_on_grid = images.permute(1, 2, 0).contiguous().view(4096, 4)
        predicted = operator.matmul(coefficients_on_grid)
        loss = criterion(predicted, voltages)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        if not np.isfinite(value):
            raise RuntimeError(f"Nonfinite loss at iteration {iteration + 1}")
        losses.append(value)
        if progress and (iteration == 0 or (iteration + 1) % 50 == 0 or iteration + 1 == config.epochs):
            print(f"Iteration {iteration + 1:4d}/{config.epochs}: voltage L1 = {value:.8g}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    optimization_seconds = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        network_images = model(noise).squeeze(0)
        coefficients = network_images.contiguous().view(4, 4096).matmul(eit2rect)
        predicted = operator.matmul(network_images.permute(1, 2, 0).contiguous().view(4096, 4))
        attention_weights = model.attention.attention_weights.detach().cpu().numpy()
        attention_scores = torch.softmax(model.attention.attention_weights, dim=-1).cpu().numpy()
        channel_weights = model.attention.channel_weights.detach().cpu().numpy()
    prediction = predicted.cpu().numpy()
    coefficients = coefficients.cpu().numpy()
    final_l1 = float(np.mean(np.abs(prediction - measurement.voltages)))
    info = {
        "case": measurement.case,
        "geometry": measurement.geometry,
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "preparation_seconds": preparation_seconds,
        "optimization_seconds": optimization_seconds,
        "initial_voltage_l1": losses[0],
        "final_voltage_l1": final_l1,
        "relative_voltage_l2": float(np.linalg.norm(prediction - measurement.voltages) / max(np.linalg.norm(measurement.voltages), np.finfo(np.float32).tiny)),
        "frequency_labels": list(measurement.frequency_labels),
    }
    arrays = {
        "coefficients": coefficients,
        "images": measurement.to_images(coefficients),
        "display_mask": measurement.display_mask,
        "network_images": network_images.cpu().numpy(),
        "observed_voltages": measurement.voltages,
        "predicted_voltages": prediction,
        "attention_weights": attention_weights,
        "attention_scores": attention_scores,
        "channel_weights": channel_weights,
        "loss": np.asarray(losses, dtype=np.float64),
        "frequency_labels": np.asarray(measurement.frequency_labels),
    }
    return arrays, info
