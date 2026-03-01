# Training GraDeT-HTR on Google Colab

Step-by-step instructions for training the GraDeT-HTR Bengali Handwritten Text Recognition model on Google Colab.

---

## Prerequisites

- A Google account with access to [Google Colab](https://colab.research.google.com/)
- Your training data (images + labels CSV), or the included sample data for testing
- (Recommended) Google Drive for storing large datasets and trained models

---

## Step 1: Open Google Colab

1. Go to [https://colab.research.google.com/](https://colab.research.google.com/)
2. Click **File > Upload notebook**
3. Upload the file `GraDeT_HTR_Colab_Training.ipynb` from this repository

---

## Step 2: Enable GPU Runtime

1. Click **Runtime > Change runtime type**
2. Select **T4 GPU** (free tier) or **A100/V100** (Colab Pro)
3. Click **Save**

> **GPU Memory Guide**:
> - T4 (16GB): Use `BATCH_SIZE = 16` to `32`
> - A100 (40GB): Use `BATCH_SIZE = 32` to `64`
> - If you get CUDA out-of-memory errors, reduce batch size

---

## Step 3: Prepare Your Training Data

### Data Format

Your training data must follow this structure:

```
your_data/
├── images/
│   ├── 1_1_1_1.jpg     # Word-level cropped images
│   ├── 1_1_1_2.jpg
│   ├── 1_1_2_1.jpg
│   └── ...
└── labels/
    └── label.csv        # CSV with image_id,text columns
```

**label.csv format** (header required):
```csv
image_id,text
1_1_1_1.jpg,কথা
1_1_1_2.jpg,প্রকাশ
1_1_2_1.jpg,বৈচিত্র্যময়
1_1_2_2.jpg,এই
```

**Image requirements**:
- Format: JPG or PNG
- Content: Single word crops (the model processes one word at a time)
- The model internally resizes to 32x128 pixels

### For Context-Aware Training (Optional)

Context-aware mode requires JSON files with word-level annotations including previous word context. Each JSON file should contain:
```json
{
    "output_path": "path/to/word_image.jpg",
    "text": "current_word_text"
}
```

### Uploading Data to Google Drive

1. Create a folder in Google Drive, e.g., `GraDeT_HTR_data/`
2. Upload your `images/` folder and `labels/` folder there
3. In the notebook, mount Google Drive and set paths accordingly

---

## Step 4: Run the Notebook

Execute cells in order. Here is what each section does:

### Cell 1 - Check GPU
Verifies that a GPU is available and shows GPU info.

### Cell 2 - Clone Repo & Install Dependencies
```bash
# Clones the repository
git clone https://github.com/shyan23/GraDeT-HTR_context.git /content/GraDeT-HTR

# Installs additional dependencies not in Colab by default
pip install marisa-trie==1.2.1
pip install git+https://github.com/csebuetnlp/normalizer@d80c3c4#egg=normalizer
pip install albumentations==2.0.5
```

> **Note**: Most dependencies (PyTorch, transformers, PIL, pandas, etc.) are already pre-installed in Colab.

### Cell 3 - Set Data Paths
Choose one of three options:
- **Option A**: Mount Google Drive (recommended for large datasets)
- **Option B**: Use included sample data (10 images, for testing the pipeline)
- **Option C**: Upload a zip file directly

### Cell 4 - Verify Data
Confirms that images and labels exist and shows sample images.

### Cell 5 - Setup Imports
Adds the project to Python path and imports all modules.

### Cell 6 - Configuration
**Modify these values before training**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONTEXT_AWARE` | `False` | Set `True` for context-aware dual-path training |
| `BATCH_SIZE` | `32` | Reduce if GPU memory is insufficient |
| `EPOCHS` | `10` | Number of training epochs |
| `LEARNING_RATE` | `1e-4` | Adam optimizer learning rate |
| `MARGIN_LAMBDA` | `0.5` | Margin loss weight (context mode only) |

### Cell 7 - Initialize Model & Data
- Creates the DTrOCRConfig with the Bengali grapheme vocabulary
- Splits data into train/validation sets (95%/5% for standard, 90%/10% for context)
- Initializes the DTrOCRLMHeadModel (~85M parameters)
- Sets up Adam optimizer with mixed precision (AMP)

### Cell 8 - (Optional) Resume from Checkpoint
Uncomment and set `CHECKPOINT_PATH` if resuming a previous training run.

### Cell 9 - Training Loop
Runs the full training loop. For each epoch:
- **Standard mode**: Single forward pass with cross-entropy loss
- **Context-aware mode**: Dual forward pass with margin penalty loss:
  `L = -log(P_ctx) - log(P_iso) + lambda * max(0, loss_ctx - loss_iso)`
- Saves a checkpoint after every epoch

### Cell 10 - Save Final Model
Saves the final model weights to `output/final_model.pth`.

### Cell 11 - Plot Training Curves
Generates loss and accuracy plots.

### Cell 12 - Download Model
Download the trained model to your local machine or copy to Google Drive.

### Cell 13 - (Optional) Quick Inference Test
Runs inference on a sample image to verify the model works.

---

## Step 5: Retrieve Your Trained Model

After training completes, you have several options:

### Download directly
The notebook's Cell 12 triggers a browser download of the `.pth` file.

### Copy to Google Drive
```python
import shutil
shutil.copy('/content/GraDeT-HTR/output/final_model.pth', '/content/drive/MyDrive/')
```

### Copy all checkpoints
```python
!cp /content/GraDeT-HTR/output/*.pt /content/drive/MyDrive/GraDeT_HTR_checkpoints/
```

---

## Using the Trained Model for Inference

After training, use the model locally with `inference.py`:

```bash
cd GraDeT-HTR

# For images
mkdir -p input_pages
# Place your page images in input_pages/ (named 1_1.jpg, 1_2.jpg, etc.)
python inference.py --weights output/final_model.pth

# For PDFs
python inference.py --weights output/final_model.pth --pdf
```

Output text files will be saved in `output_texts/`.

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` (try 16, then 8)
- Restart the runtime: **Runtime > Restart runtime**

### `ModuleNotFoundError: No module named 'normalizer'`
- Re-run the dependency installation cell
- Make sure the `pip install git+https://github.com/csebuetnlp/normalizer@d80c3c4` command succeeded

### `FileNotFoundError` for vocab file
- Verify the repository was cloned correctly
- Check that `tokenization/bn_grapheme_1296_from_bengali.ai.buet.txt` exists:
  ```python
  import os
  print(os.path.exists("/content/GraDeT-HTR/tokenization/bn_grapheme_1296_from_bengali.ai.buet.txt"))
  ```

### Colab session disconnects
- Training resumes from the last checkpoint. Upload the checkpoint and use Cell 8 to resume.
- To prevent disconnection, keep the browser tab active.

### `RuntimeError: expected scalar type Half but found Float`
- This is an AMP issue. Set `use_amp = False` in the optimizer cell and remove the `torch.autocast` blocks, or ensure all model inputs are on the same device.

### Training is very slow on CPU
- Verify GPU is enabled: **Runtime > Change runtime type > GPU**
- Run `torch.cuda.is_available()` - should return `True`

---

## Model Architecture Summary

```
DTrOCRLMHeadModel (~85M params)
├── DTrOCRModel (transformer)
│   ├── ViTPatchEmbeddings  (image → 32 patch embeddings)
│   │   └── Conv2d(3, 768, kernel=(4,8), stride=(4,8))
│   ├── token_embedding     (1420 graphemes → 768-dim)
│   ├── positional_embedding (256 positions → 768-dim)
│   ├── 12x GPT2Block       (768-dim, 12 heads, causal attention)
│   └── LayerNorm
└── language_model_head      (768 → 1420 vocab logits)
```

- **Input**: 32x128 RGB image + Bengali grapheme tokens
- **Patch embeddings**: 4x8 patches = 32 image tokens
- **Max sequence**: 256 positions (32 image + 224 text tokens)
- **Vocabulary**: 1420 Bengali graphemes (1296 graphemes + special tokens)

---

## Recommended Training Configurations

### Quick test (verify pipeline works)
```python
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 1e-4
# Use sample_train/ data
```

### Standard training (small dataset, ~10K images)
```python
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
```

### Standard training (large dataset, ~100K+ images)
```python
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 5e-5
```

### Context-aware training
```python
CONTEXT_AWARE = True
BATCH_SIZE = 16       # Needs more memory (2 forward passes)
EPOCHS = 20
LEARNING_RATE = 1e-4
MARGIN_LAMBDA = 0.5
```
