"""Render a stored MAIP reconstruction without fitting a network."""
import argparse
from pathlib import Path
from maip.visualization import render


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=Path(__file__).resolve().parent / "examples/reconstruction.npz")
    parser.add_argument("--output", type=Path, default=Path("results/visualization"))
    args = parser.parse_args()
    render(args.result, args.output)
    print(f"Figures saved to {args.output.resolve()}")
