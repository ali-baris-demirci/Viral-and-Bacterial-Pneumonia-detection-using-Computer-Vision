# Viral and Bacterial Pneumonia Detection using Computer Vision ✅

## Project Summary
This project uses chest X-ray images to classify images into **NORMAL**, **virus**, and **bacteria** classes. The main script, `demirci.py`, performs dataset organization (splitting pneumonia cases into `virus` and `bacteria`), sets up a data generator, fine-tunes a ResNet50V2 model, and evaluates the trained model with accuracy/loss plots and a classification report/confusion matrix.

---

## Repository contents
- `demirci.py` - Main script. Performs dataset restructuring, model training, evaluation, and plotting.

---

## Dataset layout (expected)
Organize your dataset under the root as follows:

```
chest_xray/
  ├─ train/
  │   ├─ NORMAL/
  │   └─ PNEUMONIA/  # images containing 'virus' or 'bacteria' in filename
  ├─ val/
  │   ├─ NORMAL/
  │   └─ PNEUMONIA/
  └─ test/
      ├─ NORMAL/
      └─ PNEUMONIA/
```

The script will create `virus` and `bacteria` folders inside each split and move files from `PNEUMONIA` based on whether the filename contains the substring `virus` or `bacteria` (case-insensitive).

---

## Requirements
- Python 3.8+
- TensorFlow (tested with 2.x)
- numpy, pandas, matplotlib, seaborn, scikit-learn

Suggested install (virtualenv):

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

(You may prefer to create a `requirements.txt` for reproducibility.)

---

## Usage
Run the main script from the repository root:

```bash
python demirci.py
```

What the script does:
- Creates `virus` and `bacteria` folders inside `chest_xray/train`, `chest_xray/val`, and `chest_xray/test` and moves images based on filename.
- Builds a custom subset generator for balanced sampling.
- Loads `ResNet50V2` (ImageNet weights), unfreezes the last 100 layers, and fine-tunes on the dataset.
- Saves the best model to `best_model.keras` and plots training curves and a confusion matrix.
- Prints a classification report and test accuracy.

---

## Configuration notes / Hyperparameters
- Image size: 512x512 (see `img_height`, `img_width` in `demirci.py`)
- Batch sizes: `train_batch_size = 32`, `val_test_batch_size = 16`
- Learning rate: 1e-5
- Steps and subset sizes are computed inside the script; adjust `subset_size` and `steps_per_epoch` as needed based on dataset size and GPU memory.

---

## Limitations & Tips 💡
- The file-moving logic expects filenames to include `virus` or `bacteria`. If your filenames differ, update the detection logic in `demirci.py`.
- Training a ResNet50V2 on high-resolution images (512x512) is resource-intensive. Use a GPU and consider lowering image size or batch size if you run out of memory.
- The `custom_subset_generator` provides an easy way to sample balanced batches; ensure your dataset has sufficient images per class for the selected `subset_size`.
- Add additional error handling (missing directories, corrupted images) for production use.

---

## Possible Improvements
- Add a CLI or config file for hyperparameters (image size, batch size, epochs, paths).
- Add unit tests for dataset processing and generator logic.
- Save training logs (TensorBoard) and more robust model checkpointing.
- Explore more recent architectures or ensemble methods for better accuracy.

---

## License
Include a license if you intend to share this project publicly (e.g., MIT). Add `LICENSE` file to the repo.

---

If you'd like, I can also:
- create a `requirements.txt` with pinned versions ✅
- add a small CLI wrapper to configure training parameters ✅

Want me to add either of those now? 🔧
