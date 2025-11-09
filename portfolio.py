# portfolio.py
from pathlib import Path
import pandas as pd

class Portfolio:
    def __init__(self, csv_path: str | None = None):
        self.path = self._resolve_path(csv_path)
        self.data = pd.read_csv(self.path)

    def _resolve_path(self, csv_path: str | None) -> Path:
        if csv_path:  # explicit path from caller
            p = Path(csv_path)
            if p.exists():
                return p

        # Try common locations relative to this file and the CWD
        here = Path(__file__).resolve().parent
        candidates = [
            here / "my_portfolio.csv",        # same folder as portfolio.py
            here.parent / "my_portfolio.csv", # repo root if this file is in a subfolder
            Path.cwd() / "my_portfolio.csv",  # working directory (Streamlit uses repo root as CWD)
        ]
        for p in candidates:
            if p.exists():
                return p

        tried = "\n  - ".join(str(p) for p in candidates + ([Path(csv_path)] if csv_path else []))
        raise FileNotFoundError(
            "Could not find my_portfolio.csv. Tried:\n  - " + tried
        )

    def load_portfolio(self):
        return self.data
