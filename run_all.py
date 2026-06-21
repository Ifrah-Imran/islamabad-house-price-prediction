"""Run full pipeline: scrape → train → launch UI."""
import subprocess
import sys


def main():
    steps = [
        ("Scraping data (300–400 listings)...", [sys.executable, "scraper.py"]),
        ("Training models...", [sys.executable, "train_models.py"]),
    ]
    for label, cmd in steps:
        print(f"\n{'='*60}\n{label}\n{'='*60}")
        subprocess.run(cmd, check=True)
    print("\nDone. Start UI with: streamlit run app.py")


if __name__ == "__main__":
    main()
