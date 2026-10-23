"""Run the pipeline end to end.

  1  download_and_process  dry-season Sentinel-2 composites per site and year
  2  features              NDVI, bare soil index and bare ground metrics
  3  make_figures          README images in docs/
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import download_and_process  # noqa: E402
import features  # noqa: E402
import make_figures  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run the SA mine site pipeline.")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], help="run one step only (default: all)")
    parser.add_argument("--years", type=int, nargs="+", help="years for step 1 (default 2017 to 2026)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.step in (None, 1):
        download_and_process.main(args.years)
    if args.step in (None, 2):
        features.main()
    if args.step in (None, 3):
        make_figures.main()


if __name__ == "__main__":
    main()
