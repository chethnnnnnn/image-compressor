"""
Image compression with Singular Value Decomposition (SVD).

Usage:
    pip install numpy pillow matplotlib
    python svd_compression.py --image test_image.png
    python svd_compression.py --image photo.jpg --ranks 5 20 50 100 --gray

If the image file does not exist, a synthetic test image is generated at that path.
Outputs (written to --out, default ./svd_output):
    - rank_<k>.png              reconstructed image for each rank k
    - comparison.png            original vs. reconstructions grid
    - singular_values.png       singular value spectrum + cumulative energy
    - summary printed to console (compression ratio, PSNR, energy kept)
"""

import argparse
import os

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_test_image(path, size=256):
    """Create a synthetic image with gradients, shapes and texture."""
    y, x = np.mgrid[0:size, 0:size] / size
    r = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * x)
    g = y
    b = 0.5 + 0.5 * np.cos(2 * np.pi * 2 * (x + y))
    img = np.stack([r, g, b], axis=-1)

    # circle
    cx, cy, rad = 0.35, 0.4, 0.18
    mask = (x - cx) ** 2 + (y - cy) ** 2 < rad ** 2
    img[mask] = [1.0, 0.85, 0.1]
    # square
    img[int(0.55 * size):int(0.85 * size), int(0.55 * size):int(0.85 * size)] = [0.1, 0.2, 0.8]
    # noise texture
    rng = np.random.default_rng(0)
    img = np.clip(img + 0.05 * rng.standard_normal(img.shape), 0, 1)

    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    print(f"Generated synthetic test image: {path}")


def load_image(path, gray):
    img = Image.open(path)
    img = img.convert("L") if gray else img.convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def svd_channels(img):
    """Return list of (U, S, Vt) per channel."""
    channels = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[2])]
    return [np.linalg.svd(ch, full_matrices=False) for ch in channels]


def reconstruct(decomps, k, is_gray):
    chans = [(U[:, :k] * S[:k]) @ Vt[:k, :] for U, S, Vt in decomps]
    out = chans[0] if is_gray else np.stack(chans, axis=-1)
    return np.clip(out, 0, 1)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def compression_ratio(h, w, k, n_channels):
    original = h * w * n_channels
    compressed = k * (h + w + 1) * n_channels
    return original / compressed


def main():
    parser = argparse.ArgumentParser(description="SVD image compression demo")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--ranks", type=int, nargs="+", default=[5, 10, 25, 50, 100],
                        help="Ranks k to keep (default: 5 10 25 50 100)")
    parser.add_argument("--gray", action="store_true", help="Convert to grayscale first")
    parser.add_argument("--out", default="svd_output", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        make_test_image(args.image)

    os.makedirs(args.out, exist_ok=True)
    img = load_image(args.image, args.gray)
    is_gray = img.ndim == 2
    h, w = img.shape[:2]
    n_ch = 1 if is_gray else img.shape[2]
    max_rank = min(h, w)
    ranks = sorted({k for k in args.ranks if 1 <= k <= max_rank})
    if not ranks:
        parser.error(f"No valid ranks; must be between 1 and {max_rank}")

    print(f"Image: {args.image}  size={w}x{h}  channels={n_ch}  max rank={max_rank}")
    decomps = svd_channels(img)

    # Energy = cumulative squared singular values, averaged across channels
    energies = [np.cumsum(S ** 2) / np.sum(S ** 2) for _, S, _ in decomps]
    mean_energy = np.mean(energies, axis=0)

    print(f"\n{'rank':>6} {'ratio':>8} {'PSNR(dB)':>9} {'energy':>8}")
    recons = {}
    for k in ranks:
        rec = reconstruct(decomps, k, is_gray)
        recons[k] = rec
        Image.fromarray((rec * 255).round().astype(np.uint8)).save(
            os.path.join(args.out, f"rank_{k}.png"))
        print(f"{k:>6} {compression_ratio(h, w, k, n_ch):>7.2f}x "
              f"{psnr(img, rec):>9.2f} {mean_energy[k - 1] * 100:>7.2f}%")

    # Comparison grid
    n = len(ranks) + 1
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    cmap = "gray" if is_gray else None
    axes[0].imshow(img, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title("Original")
    for ax, k in zip(axes[1:], ranks):
        ax.imshow(recons[k], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"k={k}  PSNR={psnr(img, recons[k]):.1f} dB\n"
                     f"ratio={compression_ratio(h, w, k, n_ch):.1f}x")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "comparison.png"), dpi=120)
    plt.close(fig)

    # Singular value spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    labels = ["Gray"] if is_gray else ["R", "G", "B", "A"][:n_ch]
    colors = ["black"] if is_gray else ["tab:red", "tab:green", "tab:blue", "gray"][:n_ch]
    for (_, S, _), lab, col in zip(decomps, labels, colors):
        ax1.semilogy(S, label=lab, color=col)
    ax1.set_title("Singular values")
    ax1.set_xlabel("index")
    ax1.legend()
    ax2.plot(np.arange(1, max_rank + 1), mean_energy * 100)
    for k in ranks:
        ax2.axvline(k, ls="--", color="gray", alpha=0.5)
    ax2.set_title("Cumulative energy retained")
    ax2.set_xlabel("rank k")
    ax2.set_ylabel("%")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "singular_values.png"), dpi=120)
    plt.close(fig)

    print(f"\nSaved outputs to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
