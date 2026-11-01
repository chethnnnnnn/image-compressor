"""
SVD Image Compressor - web app.

Upload a picture, choose how much of it to keep, and download the result.

Run (from this folder):
    python -m pip install -r requirements.txt
    python -m streamlit run app.py
(or double-click run_app.bat on Windows)
"""

import io
import os

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from svd_compression import svd_channels, reconstruct, psnr, compression_ratio, make_test_image

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "test_image.png")

st.set_page_config(page_title="SVD Image Compressor", layout="wide")


@st.cache_data(show_spinner="Running SVD on your image...", max_entries=5)  # bound memory on shared hosting
def decompose(data, gray, max_side):
    img = Image.open(io.BytesIO(data))
    img = img.convert("L") if gray else img.convert("RGB")
    img.thumbnail((max_side, max_side))  # keep SVD fast on large photos
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr, svd_channels(arr)


def to_png(arr):
    buf = io.BytesIO()
    Image.fromarray((arr * 255).round().astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def to_npz(decomps, k):
    """The truly compressed form: only the top-k factors of each channel."""
    buf = io.BytesIO()
    parts = {}
    for c, (U, S, Vt) in enumerate(decomps):
        parts[f"U{c}"] = U[:, :k].astype(np.float16)
        parts[f"S{c}"] = S[:k].astype(np.float32)
        parts[f"Vt{c}"] = Vt[:k, :].astype(np.float16)
    np.savez_compressed(buf, **parts)
    return buf.getvalue()


def kb(n):
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"


# ---------- Sidebar ----------
st.sidebar.header("Settings")
gray = st.sidebar.checkbox("Convert to grayscale", value=False)
max_side = st.sidebar.select_slider(
    "Max image size (px)", options=[256, 512, 768, 1024, 1536], value=768,
    help="Big photos are resized to this size first so the SVD stays fast.")
mode = st.sidebar.radio("Choose how much to keep", ["Target quality", "Manual rank k"])

# ---------- Input ----------
st.title("SVD Image Compressor")
st.caption("Upload a picture. The app keeps only its strongest singular values "
           "and shows how small it can get while still looking right.")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])
if st.button("Or try the sample image"):
    st.session_state.sample = True
if uploaded is not None:
    st.session_state.sample = False
    data, name = uploaded.getvalue(), uploaded.name
elif st.session_state.get("sample"):
    if not os.path.exists(SAMPLE):
        make_test_image(SAMPLE)
    with open(SAMPLE, "rb") as f:
        data, name = f.read(), "test_image.png"
else:
    st.info("Upload a PNG or JPG to get started.")
    st.stop()

img, decomps = decompose(data, gray, max_side)
is_gray = img.ndim == 2
h, w = img.shape[:2]
n_ch = 1 if is_gray else 3
max_rank = min(h, w)
energy = np.mean([np.cumsum(S ** 2) / np.sum(S ** 2) for _, S, _ in decomps], axis=0)

if mode == "Target quality":
    target = st.sidebar.slider("Energy to keep (%)", 90.0, 99.99, 99.5, step=0.01,
                               help="Higher keeps more detail but compresses less.")
    k = min(int(np.searchsorted(energy, target / 100)) + 1, max_rank)
else:
    k = st.sidebar.slider("Rank k", 1, max_rank, min(25, max_rank),
                          help="Number of singular values kept per colour channel.")

# ---------- Result ----------
rec = reconstruct(decomps, k, is_gray)
ratio = compression_ratio(h, w, k, n_ch)
quality = psnr(img, rec)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rank k", f"{k} / {max_rank}")
m2.metric("Compression", f"{ratio:.1f}×" if ratio >= 1 else "none",
          help="Pixels stored ÷ values stored in U, Σ, Vᵀ at this rank.")
m3.metric("Quality (PSNR)", f"{quality:.1f} dB",
          help="Above about 30 dB the difference is hard to see.")
m4.metric("Energy kept", f"{energy[k - 1] * 100:.2f}%")
if ratio < 1:
    st.warning("At this rank the factors take more space than the original pixels. "
               "Lower k to actually compress.")

left, right = st.columns(2)
left.image(img, caption=f"Original ({w} × {h})", clamp=True)
right.image(rec, caption=f"Rebuilt from k = {k} singular values", clamp=True)

stem = os.path.splitext(name)[0]
npz = to_npz(decomps, k)
d1, d2 = st.columns(2)
d1.download_button("Download rebuilt image (PNG)", to_png(rec),
                   file_name=f"{stem}_svd_k{k}.png", mime="image/png", width="stretch")
d2.download_button(f"Download compressed data (.npz, {kb(len(npz))})", npz,
                   file_name=f"{stem}_svd_k{k}.npz", mime="application/octet-stream",
                   width="stretch")
st.caption(f"Uploaded file: {kb(len(data))}. The .npz holds only the top-{k} factors "
           "(the real compressed form); the PNG is the image rebuilt from them.")

# ---------- Explore ----------
with st.expander("Singular value spectrum", expanded=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
    labels = ["Gray"] if is_gray else ["R", "G", "B"]
    colors = ["black"] if is_gray else ["tab:red", "tab:green", "tab:blue"]
    for (_, S, _), lab, col in zip(decomps, labels, colors):
        ax1.semilogy(S, label=lab, color=col)
    ax1.axvline(k - 1, ls="--", color="gray")
    ax1.set_title("Singular values")
    ax1.set_xlabel("index")
    ax1.legend()
    ax2.plot(np.arange(1, max_rank + 1), energy * 100)
    ax2.axvline(k, ls="--", color="gray")
    ax2.set_title("Cumulative energy retained")
    ax2.set_xlabel("rank k")
    ax2.set_ylabel("%")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with st.expander("Compare several ranks", expanded=False):
    ranks = [r for r in (5, 10, 25, 50, 100) if r <= max_rank]
    cols = st.columns(len(ranks))
    for col, r in zip(cols, ranks):
        rr = reconstruct(decomps, r, is_gray)
        col.image(rr, caption=f"k={r} · {psnr(img, rr):.1f} dB · "
                              f"{compression_ratio(h, w, r, n_ch):.1f}×", clamp=True)

with st.expander("How it works"):
    st.markdown(
        "Each colour channel is a matrix **A**. SVD factors it as **A = U Σ Vᵀ**, with the "
        "singular values in Σ sorted from strongest to weakest. Keeping only the first *k* "
        "gives the best possible rank-*k* approximation, which needs *k(m + n + 1)* numbers "
        "instead of *m · n*.\n\n"
        "A moderate *k* also smooths out fine grain and noise, because noise lives in the "
        "weak singular values. SVD cannot add detail that was never in the picture.")
