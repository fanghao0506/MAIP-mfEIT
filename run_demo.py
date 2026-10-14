"""Reconstruct one four-frequency EIT observation with MAIP."""
import argparse
import json
from pathlib import Path

import numpy as np

from maip.data import DEFAULT_DATA, available_cases, load_case
from maip.reconstruction import Config, reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case2_sim")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=950)
    parser.add_argument("--learning-rate", type=float, default=1.2e-4)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if args.list_cases:
        for name, metadata in available_cases(args.data_dir).items():
            print(f"{name:24s} {metadata['geometry']:10s} {metadata['type']}")
        return
    if (args.output / "reconstruction.npz").exists():
        parser.error("Output already contains a reconstruction; choose another --output directory")
    measurement = load_case(args.case, args.data_dir)
    print(f"Case: {args.case}; geometry: {measurement.geometry}; voltages: {measurement.voltages.shape}", flush=True)
    arrays, info = reconstruct(measurement, Config(args.epochs, args.learning_rate, args.seed), args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "reconstruction.npz", **arrays)
    np.savetxt(args.output / "conductivity.txt", arrays["coefficients"], fmt="%.9g")
    np.savetxt(args.output / "loss.csv", np.column_stack((np.arange(1, args.epochs + 1), arrays["loss"])), delimiter=",", header="iteration,voltage_l1", comments="", fmt=("%d", "%.10g"))
    (args.output / "run.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    if not args.no_figures:
        from maip.visualization import render
        render(args.output / "reconstruction.npz", args.output)
    print(f"Voltage L1: {info['initial_voltage_l1']:.6g} -> {info['final_voltage_l1']:.6g}; optimization: {info['optimization_seconds']:.2f} s", flush=True)
    print(f"Saved to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
