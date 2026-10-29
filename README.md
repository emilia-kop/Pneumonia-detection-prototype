# Pneumonia Detection from Chest X-ray Images (Educational Prototype)

> **Not a medical device.** For learning and demonstration only. Never use it to make clinical decisions.

## 1. Overview and architecture
A Streamlit web app that takes a chest X-ray (JPG/JPEG/PNG), runs a **pretrained DenseNet121** and
reports `NORMAL` or `PNEUMONIA`.

```
Upload image -> validate (Pillow) -> preprocess -> DenseNet121 (18 outputs) -> take "Pneumonia" score
             -> score >= threshold ? PNEUMONIA : NORMAL
```
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (upload, preview, Predict, result, disclaimers, error messages) |
| `pneumonia_model.py` | Loads the model once, preprocessing, prediction, error handling (renamed from `model.py`, see note below) |
| `evaluate.py` | Accuracy / precision / recall / F1 / ROC-AUC / confusion matrix on labelled folders |
| `tests/test_basic.py` | Unit tests (labels, image validation, metrics, preprocessing shape) |
| `report/` | `dataset_description.md`, `literature_survey.md`, `evaluation.md` |

## 2. Model source and licence
- **Library:** TorchXRayVision - https://github.com/mlmed/torchxrayvision (Cohen et al., MIDL 2022, arXiv:2111.00595)
- **Weights (default):** `densenet121-res224-all` (trained on NIH, PadChest, CheXpert, MIMIC-CXR, Google-NIH, OpenI, RSNA data)
- **Alternative:** `densenet121-res224-rsna` (trained on the RSNA Pneumonia Detection Challenge data)
- **Where the weights come from:** the installed library (v1.5.7, checked in its source) downloads them from the
  official release assets of https://github.com/mlmed/torchxrayvision (release `v1`), about 28 MB, into
  `~/.torchxrayvision/models_data/`. (An earlier version of this README said Hugging Face; that was wrong for this
  library version and the `from_hf_hub` argument the old code used does not exist in it.)
- **Licence:** the repository's `LICENSE` file lists the main library, including `xrv.models`, under Apache-2.0.
  It does not give the weight files a separate licence statement. **We do not bundle the weights in this ZIP**; the
  app downloads them from the authors' release page. If you ever redistribute them, re-check the repository first.
- **Architecture / input:** DenseNet121, 224x224, 1 channel, pixel range [-1024, 1024].
- **Output:** 18 values; for the weights above the one named `Pneumonia` is at index 8. The library applies a sigmoid
  and rescales so that 0.5 is the model's own operating point for that finding. It is **not** a probability.
- **ImageNet vs pneumonia-trained:** these weights are trained on chest X-ray datasets, **not** just ImageNet
  weights with a new head. We did **not** train or fine-tune anything ourselves.
- **Honest caveats:**
  1. It is a *multi-label* model (18 findings). We read the "Pneumonia" output and apply a threshold
     (default 0.5, adjustable in the sidebar). "NORMAL" = *pneumonia not flagged*, not "healthy".
  2. The displayed number is the **raw pneumonia score**, not a calibrated probability or confidence.
  3. Training data was mostly adult X-rays; the commonly used Kaggle/Kermany set is paediatric, so
     expect weaker results there (see `report/dataset_description.md`).

## 3. Getting the model weights
Nothing to do by hand. On the first prediction the app downloads the weights (about 28 MB, needs internet and access
to github.com once) and caches them in `C:\Users\<you>\.torchxrayvision\models_data\`. **No weights are bundled in this ZIP.**
To pre-download: `python -c "from pneumonia_model import PneumoniaClassifier; PneumoniaClassifier()"`
If the download is blocked, the app shows an error instead of a prediction. You can then fetch the file manually
(the library prints the exact `wget` command with the URL) and place it in the folder above.

> **Why `model.py` was renamed:** TorchXRayVision internally imports a package called `model`. Our `model.py`
> shadowed it, so `import torchxrayvision` could fail with `No module named 'model.utils'`. The file is now `pneumonia_model.py`.

## 4. Windows installation
Tested here on Python 3.13 (Linux). Python 3.10-3.12 should also work but was not tested here. Check with `python --version`.
```bat
cd pneumonia-detection
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
(`torch` is a large download. CPU-only is fine: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
before the last command.)

## 5. Run
```bat
python -m streamlit run app.py   :: opens http://localhost:8501
python -m pytest -v              :: run tests
python evaluate.py --data-dir chest_xray\test   :: evaluate on your labelled test images
```
Use a real chest X-ray image for the demo (e.g. from the dataset in `report/dataset_description.md`).

## 6. Troubleshooting
| Problem | Fix |
|---|---|
| `Required packages are missing` | Activate the venv and run `pip install -r requirements.txt` |
| `Could not load weights ...` | Check internet access to github.com (first run downloads ~28 MB); retry; check firewall/proxy, or place the file manually (section 3) |
| `streamlit` not recognised | Use `python -m streamlit run app.py` |
| Very slow first prediction | Model download + load happens once, later predictions are fast |
| Prediction looks wrong on a dataset | Likely domain shift (adult-trained weights); see `report/evaluation.md` |
| `Invalid image` | File isn't a real JPG/PNG, or is smaller than 64 px |
| Package version conflicts | Use a fresh venv; if `torchxrayvision` fails, note the error text and its version pin |

## 7. Deploy: GitHub + Streamlit Community Cloud
1. Create a GitHub repo and push (`.gitignore` already excludes weights and datasets):
   ```bat
   git init
   git add .
   git commit -m "Pneumonia detection prototype"
   git branch -M main
   git remote add origin https://github.com/<your-username>/pneumonia-detection.git
   git push -u origin main
   ```
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, choose the repo,
   branch `main`, main file `app.py`, then **Deploy**.
3. In *Advanced settings* choose a supported Python version (3.11 or 3.12 is a safe pick).
4. **PyTorch size warning (important):** on Linux, a plain `pip install torch` pulls CUDA libraries (several GB;
   we observed that here). For Streamlit Community Cloud put this as the **first line** of `requirements.txt` so the
   small CPU build is used: `--extra-index-url https://download.pytorch.org/whl/cpu`.
   **This was NOT tested here** (the sandbox could not reach that host), so verify it on your deployment.
5. **Hosting limitations:** the free tier has limited memory and CPU. Weights are downloaded from GitHub at runtime
   (not stored in the repo), so the first request after a restart is slow. If the build runs out of resources, demo locally.
   **Run it locally and confirm it works before depending on the deployed link.**

## 8. Test status (what was actually run)
Run on Linux, Python 3.13.18, torch 2.14.1, torchvision 0.29.1, torchxrayvision 1.5.7, streamlit 1.66.1.

| Check | Result |
|---|---|
| Python syntax of all files | Passed |
| `pytest`: 11 tests (labels, image validation, metrics maths, preprocessing, model errors, real-weights checks) | 11 passed |
| Real weights load (`densenet121-res224-all` and `-rsna`) | Passed (downloaded from the official GitHub release) |
| `Pneumonia` output name/index matches the label shown | Passed (index 8 for both weight files) |
| Our preprocessing vs the library's documented reference pipeline, on 4 real X-rays | Identical scores (difference 0.00000) |
| Streamlit server starts, `/_stcore/health` returns ok | Passed |
| App script runs without exceptions; upload then Predict with real X-rays (via Streamlit `AppTest`) | Passed |
| Invalid-image upload and missing-weights error messages | Passed (clear error, no prediction shown) |
| `evaluate.py` runs end to end, skips unreadable files, errors cleanly on a missing folder | Passed (mechanics only, see `report/evaluation.md`) |
| Real browser file-upload widget, screenshots | **Not tested** (no browser in the test environment) |
| Accuracy / precision / recall on a labelled pneumonia dataset | **Not done** (dataset not available offline; see `report/evaluation.md`) |
| Streamlit Community Cloud deployment | **Not tested** |

**Honest observation:** on the one known-pneumonia image we had (a COVID-19 pneumonia radiograph from the library's test
folder) both weight files gave a low `Pneumonia` score, so it was labelled NORMAL at threshold 0.5. This is a single
image, not an evaluation, but it shows the model can miss pneumonia, which is why the app shows all model outputs and a disclaimer.

## 9. Limitations
Educational prototype; not clinically validated; single-image, frontal chest X-rays only; sensitive to
image source, population and threshold; no explanation/heatmap; no results from our own evaluation until
`evaluate.py` is run.

## 10. Credits
TorchXRayVision (Apache-2.0) by Cohen et al.; dataset by Kermany et al. (CC BY 4.0). See `report/literature_survey.md`.
