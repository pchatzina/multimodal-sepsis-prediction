import sys
import subprocess
from src.utils.config import Config
from src.utils.download import download_with_wget


def main():

    # First python script that touches the file system, initialize folders
    Config.check_dirs()

    print(">>> Starting Metadata Downloads...")

    # ---------------------------------------------------------
    # 1. Download CXR Metadata
    # ---------------------------------------------------------
    print("\n[1/2] Downloading CXR metadata CSVs...")
    if hasattr(Config, "CXR_METADATA_FILES"):
        for filename, base_url in Config.CXR_METADATA_FILES:
            local_path = Config.RAW_CXR_IMG_DIR / filename
            url = f"{base_url}{filename}"

            try:
                download_with_wget(
                    url, local_path, Config.PHYSIONET_USER, Config.PHYSIONET_PASS
                )
                print(f"SUCCESS: {local_path.name}")
            except subprocess.CalledProcessError:
                print(f"CRITICAL ERROR: Failed to download {url}", file=sys.stderr)
                sys.exit(1)
    else:
        print("WARNING: Config.CXR_METADATA_FILES is missing.", file=sys.stderr)

    # ---------------------------------------------------------
    # 2. Download ECG Metadata
    # ---------------------------------------------------------
    print("\n[2/2] Downloading ECG metadata CSVs...")
    if hasattr(Config, "ECG_METADATA_FILES"):
        for filename in Config.ECG_METADATA_FILES:
            local_path = Config.RAW_ECG_DIR / filename
            url = f"{Config.URL_ECG_BASE}{filename}"

            try:
                download_with_wget(
                    url, local_path, Config.PHYSIONET_USER, Config.PHYSIONET_PASS
                )
                print(f"SUCCESS: {local_path.name}")
            except subprocess.CalledProcessError:
                print(f"CRITICAL ERROR: Failed to download {url}", file=sys.stderr)
                sys.exit(1)
    else:
        print("WARNING: Config.ECG_METADATA_FILES is missing.", file=sys.stderr)

    print("\n>>> Metadata download complete.")


if __name__ == "__main__":
    main()
