# SVD Image Compressor

Compresses images with Singular Value Decomposition (SVD): each colour channel
is factored as A = U Σ Vᵀ and only the strongest k singular values are kept.

**Try it live:** https://chethnnnnnn.github.io/image-compressor/ — runs `app.py`
entirely in your browser (the first load takes about 20–30 seconds).

## Run the app (upload a picture in your browser)

First extract the zip (right-click it, then Extract All). Then either:

**Windows:** double-click `run_app.bat` inside the extracted folder. It installs
the requirements and opens http://localhost:8501.

**From a terminal**, inside the extracted folder:

```
cd path\to\SVD_Image_Compressor
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open http://localhost:8501 in your browser.

Use `python -m pip` and `python -m streamlit` rather than plain `pip` and
`streamlit`: on many Windows installs the `streamlit` command is not on PATH
and fails with "streamlit is not recognized".

In the app:
1. Upload a PNG/JPG (or click **Or try the sample image**).
2. Pick a target quality (energy %) or set the rank k by hand in the sidebar.
3. Compare the original and the rebuilt image, and read the compression ratio,
   PSNR and energy kept.
4. Download the rebuilt PNG, or the compressed factors as `.npz`.

## Command-line version

```
python svd_compression.py --image test_image.png
python svd_compression.py --image photo.jpg --ranks 5 20 50 --gray
```

Writes `rank_k.png`, `comparison.png` and `singular_values.png` to `svd_output/`.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `index.html` | Runs `app.py` in the browser for the live link (via stlite) |
| `svd_compression.py` | SVD functions + command-line tool |
| `run_app.bat` | One-click launcher for Windows |
| `requirements.txt` | numpy, pillow, matplotlib, streamlit |
| `.streamlit/config.toml` | Starts the app without Streamlit's first-run email prompt |
| `test_image.png` | Sample image |
| `SVD_Image_Compression.pptx` | Project presentation |

## Note

SVD makes an image *smaller*, not sharper. A moderate k also smooths out
grain and noise (noise lives in the weak singular values), but it cannot add
detail that was never in the picture.
