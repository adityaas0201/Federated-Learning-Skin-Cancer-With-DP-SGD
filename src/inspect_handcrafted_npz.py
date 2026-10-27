import numpy as np
from pathlib import Path

# Update this path if your project is located elsewhere
FEATURE_DIR = Path("outputs/handcrafted_features/scaled")

FILES = [
    "train_features_scaled.npz",
    "val_features_scaled.npz",
    "test_features_scaled.npz",
]

for filename in FILES:
    path = FEATURE_DIR / filename

    print("\n" + "=" * 65)
    print(f"FILE: {filename}")
    print("=" * 65)

    if not path.exists():
        print(f"File not found: {path.resolve()}")
        continue

    # Only load NPZ files you trust
    with np.load(path, allow_pickle=False) as data:
        print("Keys:", data.files)

        for key in data.files:
            arr = data[key]
            print(f"\nKey: {key}")
            print("Shape:", arr.shape)
            print("Data type:", arr.dtype)

            if np.issubdtype(arr.dtype, np.number):
                print("Min:", np.nanmin(arr))
                print("Max:", np.nanmax(arr))
                print("Mean:", np.nanmean(arr))
                print("Std:", np.nanstd(arr))

                if arr.ndim == 2:
                    print("Number of samples:", arr.shape[0])
                    print("Number of features:", arr.shape[1])

                    if arr.shape[1] == 1320:
                        print("Confirmed: 1,320 features per sample")