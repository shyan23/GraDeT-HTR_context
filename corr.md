## Plan: Sentence-Grouped Data Generation + Context-Aware Training Fix

---

### Context

The current `generate_training_data.py` randomly shuffles all words, which destroys sentence structure. As a result:

* Context-aware training becomes meaningless.
* `ContextAwareDataset` uses `.shift(1)` to get the “previous word.”
* After shuffling, the previous word often comes from a completely unrelated sentence.
* The EMNLP paper uses a **two-stage pretraining pipeline** (line-level → word-level) before fine-tuning.

We need to:

1. Fix generation to preserve sentence order and add `line_id` markers.
2. Fix the dataset to respect sentence boundaries.
3. Add line-level generation support.
4. Generate 50k word images for context-aware training experiments.
5. Update the training notebook for multi-stage training.

---

## Files to Modify

| File                                    | Change                                                                                                         |                                                               |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `GlyphScribe/generate_training_data.py` | Add `--mode word                                                                                               | line`, preserve sentence grouping, inject `line_id` into JSON |
| `GraDeT_HTR/dataset.py`                 | Fix `_build_dataframe_from_json` to use `line_id` boundaries; update `split_context_data` to split by sentence |                                                               |
| `GlyphScribe/Makefile`                  | Add `generate-lines` and `generate-50k` targets                                                                |                                                               |
| `GraDeT_HTR_Colab_Training.ipynb`       | Add multi-stage config and stage-aware checkpoint loading                                                      |                                                               |

---

# Step 1: `GlyphScribe/generate_training_data.py`

---

### 1a. Add `--mode` Argument

```python
parser.add_argument(
    '--mode',
    type=str,
    default='word',
    choices=['word', 'line'],
    help='Generation mode: word images or line images'
)
```

---

### 1b. Replace Word Extraction with Sentence-Preserving Extraction

**Current issue:** Words are shuffled globally.

**Fix:**

* Extract full sentences while preserving grouping:

  ```python
  all_sentences = [(sentence_text, [word1, word2, ...])]
  ```

* Shuffle at **sentence level**, not word level.

* For `word` mode:

  * Build flat list of:

    ```
    (line_id, word_index, word)
    ```

  * Preserve within-sentence order.

* For `line` mode:

  * Build list of:

    ```
    (line_id, sentence_text)
    ```

---

### 1c. Inject `line_id` into JSON After Image Generation

After each image is generated:

* Re-open its JSON file.
* Add:

  * `line_id`
  * `word_index`

This ensures backward compatibility while enabling context-aware training.

---

### 1d. Update CSV Format

Add a new column:

```
image_id,text,line_id
```

Backward compatibility is preserved because `HandwrittenDataset` only reads:

* `image_id`
* `text`

---

# Step 2: `GraDeT_HTR/dataset.py`

---

### 2a. `_build_dataframe_from_json()` — Respect Sentence Boundaries

Current issue:

* Uses blind `.shift(1)` for `prev_text`.

Fix:

* Read `line_id` from JSON (default to `-1` for backward compatibility).
* When constructing `prev_text`:

  * Reset to `""` if `line_id` changes.
  * Replace `.shift(1)` with boundary-aware logic.

Goal:

* First word in each sentence → empty `prev_text`
* No cross-sentence context bleeding

---

### 2b. `split_context_data()` — Split by Sentence

Current issue:

* Splits by row.
* Words from same sentence may land in different splits.

Fix:

1. Get unique `line_id` values.
2. Split those into train/test.
3. Keep all words from same sentence in same split.
4. Rebuild `prev_text` after splitting using a helper:

```python
_rebuild_prev_text()
```

---

# Step 3: `GlyphScribe/Makefile`

Add new targets:

* `generate-50k`

  * Word mode
  * 50,000 images
* `generate-lines`

  * Line mode generation

Keep existing:

* `train`
* `train-context`

---

# Step 4: Notebook Updates (`GraDeT_HTR_Colab_Training.ipynb`)

---

### Add Training Stages

```python
TRAINING_STAGE = 1  # 1=line pretrain, 2=word pretrain, 3=finetune
```

---

### Stage-Aware Learning Rates

* **Pretraining:** `1e-4`
* **Fine-tuning:** `5e-6`
  (As specified in the EMNLP paper)

---

### Stage-Aware Checkpoint Loading

* Stage 2 loads final model from Stage 1.
* Stage 3 loads final model from Stage 2.

---

# Step 5: Generate 50k Data

1. Activate GlyphScribe virtual environment.
2. Run generation (word mode, 50k images).
3. Zip output directory.
4. Upload to Google Drive.

---

# Verification Checklist

---

### 1. Small Test Batch (100 Images)

Verify:

* JSON files contain:

  * `line_id`
  * `word_index`
* Words from same sentence share same `line_id`.
* CSV contains `line_id` column.

---

### 2. Load with `ContextAwareDataset`

Verify:

* First word of each sentence → empty `prev_text`.
* Subsequent words → correct previous word.
* No cross-sentence context bleeding.

---

### 3. Full 50k Generation

* Run generation.
* Zip dataset.
* Upload to Google Drive.
* Confirm integrity before training.

---

## Expected Outcome

After implementation:

* Sentence structure is preserved.
* Context-aware training becomes meaningful.
* No leakage across sentence boundaries.
* Multi-stage training pipeline aligns with EMNLP methodology.
* Dataset supports both word-level and line-level pretraining.

This restores architectural correctness and enables proper evaluation of context-aware improvements.
