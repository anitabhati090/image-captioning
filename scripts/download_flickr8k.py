"""Download Flickr8k from its public Kaggle mirror (requires Kaggle credentials)."""
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/flickr8k"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", "adityajn105/flickr8k", "-p", str(args.output), "--unzip"], check=True)
    print(f"Downloaded Flickr8k images and captions into {args.output}. Training uses first --limit images.")


if __name__ == "__main__":
    main()
