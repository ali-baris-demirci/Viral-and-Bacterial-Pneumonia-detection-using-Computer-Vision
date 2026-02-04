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
This project depends on the following packages (example versions shown). You can install them with pip or use the provided `requirements.txt` for reproducibility.

- Python 3.8+
- tensorflow>=2.10,<3
- numpy>=1.23
- pandas>=1.5
- matplotlib>=3.5
- seaborn>=0.12
- scikit-learn>=1.1

Install via pip (example):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Example `requirements.txt` content:

```
tensorflow>=2.10,<3
numpy>=1.23
pandas>=1.5
matplotlib>=3.5
seaborn>=0.12
scikit-learn>=1.1
```

(You can fine-tune versions for your environment or GPU support.)

---

## Usage
Run the main script from the repository root:

```bash
python demirci.py
```

**Note:** This project was developed to be run on Google Colab using Google Drive for dataset storage. In Colab, mount your Drive (for example `from google.colab import drive; drive.mount('/content/drive')`) and place the `chest_xray` directory in Drive, or set `base_dir` accordingly (e.g., `/content/drive/MyDrive/`).

What the script does:
- Creates `virus` and `bacteria` folders inside `chest_xray/train`, `chest_xray/val`, and `chest_xray/test` and moves images based on filename.
- Builds a custom subset generator for balanced sampling.
- Loads `ResNet50V2` (ImageNet weights), unfreezes the last 100 layers, and fine-tunes on the dataset.
- Saves the best model to `best_model.keras` and plots training curves and a confusion matrix.
- Prints a classification report and test accuracy.

## Running on Google Colab / Google Drive
This project was designed to be used on **Google Colab** with the dataset stored on **Google Drive**. The repository does not include the dataset (it must be obtained separately). Example Colab snippet:

```python
from google.colab import drive
drive.mount('/content/drive')

# Option 1: copy the dataset into the Colab runtime
!cp -r /content/drive/MyDrive/path/to/chest_xray /content/

# Option 2: run directly from Drive - set base_dir in demirci.py accordingly
# base_dir = '/content/drive/MyDrive/path/to/project'
```

Make sure the dataset follows the expected `chest_xray/` layout described above and set `base_dir` appropriately when running on Colab or Drive.

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

## License
This project is licensed under the MIT License — see the `LICENSE` file for the full text.

MIT License (SPDX: MIT)

Copyright (c) 2026 ali-baris-demirci

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---


