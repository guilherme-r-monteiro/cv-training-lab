import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = Path("data")


def convert(name: str) -> None:
    frame = pd.read_parquet(DATA_DIR / f"{name}.parquet")
    images = np.stack([np.asarray(Image.open(io.BytesIO(row["bytes"])).convert("RGB")) for row in frame["img"]])
    np.savez(DATA_DIR / f"{name}.npz", images=images, labels=frame["label"].to_numpy())


if __name__ == "__main__":
    convert("train")
    convert("test")
