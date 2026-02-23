import sys
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "pipeline.log"),
    ],
)
log = logging.getLogger(__name__)

(ROOT / "logs").mkdir(exist_ok=True)


def run_step_1():
    log.info("=" * 60)
    log.info("STEP 1: Data Download & Preprocessing")
    log.info("=" * 60)
    from scripts import download_and_process as s1
    s1.main()


def run_step_2():
    log.info("=" * 60)
    log.info("STEP 2: Feature Engineering")
    log.info("=" * 60)
    from scripts import features as s2
    s2.main()


def run_step_3():
    log.info("=" * 60)
    log.info("STEP 3: Model Training")
    log.info("=" * 60)
    from scripts import train as s3
    s3.main()


def run_step_4():
    log.info("=" * 60)
    log.info("STEP 4: Inference")
    log.info("=" * 60)
    from scripts import predict as s4
    # Predict all sites
    import importlib
    s1 = importlib.import_module("scripts.download_and_process")
    for site in s1.SA_SITES:
        try:
            s4.predict_site(site)
        except Exception as e:
            log.error(f"  Error on {site}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run SA Mining Analyzer pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Run a specific step only (1-4). Default: all steps.")
    args = parser.parse_args()

    # Rename scripts for import
    import importlib.util, types

    def load_script(filename, module_name):
        spec = importlib.util.spec_from_file_location(
            module_name, ROOT / "scripts" / filename
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    if args.step is None or args.step == 1:
        s1 = load_script("01_download_and_process.py", "step1")
        s1.main()

    if args.step is None or args.step == 2:
        s2 = load_script("02_features.py", "step2")
        s2.main()

    if args.step is None or args.step == 3:
        s3 = load_script("03_train.py", "step3")
        s3.main()

    if args.step is None or args.step == 4:
        s4 = load_script("04_predict.py", "step4")
        s1_mod = sys.modules.get("step1")
        if s1_mod:
            for site in s1_mod.SA_SITES:
                try:
                    s4.predict_site(site)
                except Exception as e:
                    log.error(f"Error on {site}: {e}")
        else:
            log.warning("Could not load site list for inference; loading step 1 module.")
            s1_fresh = load_script("01_download_and_process.py", "step1_b")
            for site in s1_fresh.SA_SITES:
                try:
                    s4.predict_site(site)
                except Exception as e:
                    log.error(f"Error on {site}: {e}")

    log.info("=" * 60)
    log.info("Pipeline complete. Launch dashboard with: streamlit run app.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()