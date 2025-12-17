# Context-Aware GraDeT-HTR: Detailed Implementation Guide

This document provides a comprehensive technical explanation of all code changes made to implement the context-aware extension with dual-path loss and difficulty-aware weighting.

---

## Table of Contents
1. [data.py - Data Structure Extensions](#1-datapy---data-structure-extensions)
2. [processor.py - Context Token Handling](#2-processorpy---context-token-handling)
3. [model.py - Architecture Modifications](#3-modelpy---architecture-modifications)
4. [dataset.py - Context-Aware Dataset](#4-datasetpy---context-aware-dataset)
5. [train.py - Dual-Path Training Loop](#5-trainpy---dual-path-training-loop)
6. [Complete Flow Diagram](#6-complete-flow-diagram)

---

## 1. data.py - Data Structure Extensions

### Changes Made

**Added two new fields to existing dataclasses:**

```python
@dataclass
class DTrOCRLMHeadModelOutput:
    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
    accuracy: Optional[torch.FloatTensor] = None
    past_key_values: Optional[torch.FloatTensor] = None
    per_sample_loss: Optional[torch.FloatTensor] = None  # NEW FIELD

@dataclass
class DTrOCRProcessorOutput:
    pixel_values: Optional[torch.FloatTensor] = None
    input_ids: Optional[Union[torch.LongTensor, np.ndarray, List[int]]] = None
    attention_mask: Optional[Union[torch.FloatTensor, np.ndarray, List[int]]] = None
    labels: Optional[Union[torch.LongTensor, np.ndarray, List[int]]] = None
    context_length: Optional[int] = None  # NEW FIELD
```

### Purpose

1. **`per_sample_loss`**: Enables difficulty-aware weighting by providing individual loss values for each sample in the batch
2. **`context_length`**: Tracks how many tokens at the start of input_ids are context (including SEP), needed for proper sequence reordering and loss computation

---

## 2. processor.py - Context Token Handling

### Complete Implementation

The processor now accepts a `context_text` parameter and handles tokenization + concatenation:

```python
def __call__(
    self,
    images: Union[Image.Image, List[Image.Image]] = None,
    texts: Union[str, List[str]] = None,
    context_text: Union[str, List[str]] = None,  # NEW PARAMETER
    return_labels: bool = False,
    input_data_format: str = 'channels_last',
    padding: Union[bool, str] = False,
    *args,
    **kwargs
) -> DTrOCRProcessorOutput:
```

### Step-by-Step Processing

#### Step 1: Initialize Context Length
```python
import torch
context_length = 0
```

#### Step 2: Tokenize Target Text
```python
text_inputs = self.tokeniser(
    texts, padding=padding
) if texts is not None else None
```

Output: `{'input_ids': [BOS, target_token_1, target_token_2, ...], 'attention_mask': [1, 1, 1, ...]}`

#### Step 3: Handle Context (If Provided)

**3a. Tokenize Context:**
```python
if context_text is not None and context_text != "" and text_inputs is not None:
    context_inputs = self.tokeniser(
        context_text, padding=False
    )
    sep_token_id = self.tokeniser.eos_token_id  # Reuse EOS as [SEP]
```

**3b. Extract Context IDs (Handle List/Tensor):**
```python
context_ids = context_inputs['input_ids']

# Handle nested lists or tensors
if isinstance(context_ids, list):
    if len(context_ids) > 0 and isinstance(context_ids[0], list):
        context_ids = context_ids[0]
elif hasattr(context_ids, 'tolist'):
    context_ids = context_ids.tolist()
    if isinstance(context_ids[0], list):
        context_ids = context_ids[0]
```

**3c. Create Context with Separator:**
```python
context_with_sep = context_ids + [sep_token_id]
context_length = len(context_with_sep)
```

Example: If context is "আমার" → tokenized to [t1, t2] → with SEP: [t1, t2, SEP] → `context_length = 3`

**3d. Extract Target IDs:**
```python
target_ids = text_inputs['input_ids']
# Handle list/tensor conversion (same as context)
if hasattr(target_ids, 'tolist'):
    target_ids = target_ids.tolist()
    if isinstance(target_ids[0], list):
        target_ids = target_ids[0]
```

**3e. Concatenate: Context + Target:**
```python
combined_ids = context_with_sep + target_ids
text_inputs['input_ids'] = torch.tensor([combined_ids])
```

**Result:** `[ctx_t1, ctx_t2, SEP, BOS, target_t1, target_t2, ...]`

**3f. Update Attention Mask:**
```python
context_mask = [1] * context_length  # All context tokens attended
target_mask = text_inputs['attention_mask']  # Extract target mask
# Handle list/tensor...
combined_mask = context_mask + target_mask
text_inputs['attention_mask'] = torch.tensor([combined_mask])
```

#### Step 4: Process Images
```python
image_inputs = self.vit_processor(
    images, input_data_format=input_data_format, *args, **kwargs
) if images is not None else None
```

#### Step 5: Return Output
```python
return DTrOCRProcessorOutput(
    pixel_values=image_inputs["pixel_values"] if images is not None else None,
    input_ids=text_inputs['input_ids'] if texts is not None else None,
    attention_mask=text_inputs['attention_mask'] if texts is not None else None,
    labels=text_inputs['input_ids'] if texts is not None and return_labels else None,
    context_length=context_length  # Pass context length to model
)
```

### Example Flow

**Input:**
- Image: word image of "বাংলা"
- Text: "বাংলা"
- Context: "আমার" (previous word)

**Processing:**
1. Tokenize "আমার" → [15, 23] (hypothetical token IDs)
2. Add SEP → [15, 23, 3] (assuming EOS token ID is 3)
3. Tokenize "বাংলা" → [1, 45, 67, 2] (BOS=1, tokens, EOS=2)
4. Concatenate → [15, 23, 3, 1, 45, 67, 2]
5. `context_length = 3`

**Output:**
```python
DTrOCRProcessorOutput(
    pixel_values=<image tensor>,
    input_ids=[15, 23, 3, 1, 45, 67, 2],
    attention_mask=[1, 1, 1, 1, 1, 1, 1],
    labels=[15, 23, 3, 1, 45, 67, 2],
    context_length=3
)
```

---

## 3. model.py - Architecture Modifications

### Overview of Changes

Three major modifications:
1. **DTrOCRModel.forward**: Reorder embeddings to place context before images
2. **Attention mask adjustment**: Handle the new sequence structure
3. **DTrOCRLMHeadModel.forward**: Per-sample loss computation with correct skip positions

---

### 3.1 DTrOCRModel.forward - Embedding Reordering

#### New Parameter
```python
def forward(
    self,
    pixel_values: torch.Tensor,
    input_ids: torch.LongTensor,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = False,
    context_length: Optional[int] = 0,  # NEW PARAMETER
) -> DTrOCRModelOutput:
```

#### Embedding Generation (Lines 72-73)
```python
patch_embeddings = self.patch_embeddings(pixel_values) if past_length == 0 else None
token_embeddings = self.token_embedding(input_ids)
```

- **patch_embeddings**: `[batch, 128, hidden_size]` (128 image patches)
- **token_embeddings**: `[batch, seq_len, hidden_size]` where seq_len includes context + target

#### Critical Reordering Logic (Lines 75-92)

**Case 1: Context Mode (context_length > 0)**

```python
if patch_embeddings is not None and context_length > 0:
    # Split token embeddings into context and target parts
    context_embeddings = token_embeddings[:, :context_length, :]
    # Shape: [batch, context_length, hidden_size]

    target_embeddings = token_embeddings[:, context_length:, :]
    # Shape: [batch, target_length, hidden_size]

    # Concatenate: [context, SEP] + [IMAGE patches] + [BOS, target tokens]
    patch_and_token_embeddings = torch.concat([
        context_embeddings,   # [batch, ctx_len, hidden]
        patch_embeddings,     # [batch, 128, hidden]
        target_embeddings     # [batch, tgt_len, hidden]
    ], dim=-2)
```

**Result:** `[batch, ctx_len + 128 + tgt_len, hidden_size]`

**Sequence structure:** `[ctx_t1, ctx_t2, SEP, img_patch_0, img_patch_1, ..., img_patch_127, BOS, tgt_t1, tgt_t2, ...]`

**Case 2: No Context (standard mode)**

```python
elif patch_embeddings is not None:
    # Standard: [IMAGE patches] + [BOS, target tokens]
    patch_and_token_embeddings = torch.concat([patch_embeddings, token_embeddings], dim=-2)
```

**Case 3: Using Cache (past_key_values)**

```python
else:
    # Generation mode with KV cache
    patch_and_token_embeddings = token_embeddings
```

---

### 3.2 Attention Mask Reordering (Lines 106-134)

The attention mask must match the reordered sequence structure.

#### Context Mode Adjustment

```python
if attention_mask is not None:
    if context_length > 0 and patch_embeddings is not None:
        # Original attention_mask: [batch, ctx_len + tgt_len]
        # We need: [batch, ctx_len + 128 + tgt_len]

        # Split into context and target masks
        context_mask = attention_mask[:, :context_length]
        # Shape: [batch, ctx_len]

        target_mask = attention_mask[:, context_length:]
        # Shape: [batch, tgt_len]

        # Create mask for image patches (all 1s - always attend)
        patch_mask = torch.ones(
            attention_mask.shape[0],      # batch_size
            patch_embeddings.shape[-2],   # 128
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        # Shape: [batch, 128]

        # Concatenate: [context_mask, patch_mask, target_mask]
        attention_mask = torch.concat([context_mask, patch_mask, target_mask], dim=-1)
        # Final shape: [batch, ctx_len + 128 + tgt_len]
```

**Example:**
- Original input_ids: [ctx1, ctx2, SEP, BOS, tgt1, tgt2] → length 6
- Original attention_mask: [1, 1, 1, 1, 1, 1]
- After reordering with images (128 patches):
  - context_mask: [1, 1, 1] (first 3 tokens)
  - patch_mask: [1, 1, 1, ..., 1] (128 ones)
  - target_mask: [1, 1, 1] (last 3 tokens)
  - Final: [1, 1, 1, 1, ..., 1, 1, 1, 1] → length 134

#### Standard Mode (No Context)

```python
else:
    # Standard: prepend patch mask to existing attention mask
    attention_mask = torch.concat(
        [
            torch.ones(
                attention_mask.shape[0],
                patch_embeddings.shape[-2] if patch_embeddings is not None else past_length,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            ),
            attention_mask
        ], dim=-1
    )
```

#### Causal Attention Mask Preparation

```python
if self._attn_implementation == "flash_attention_2":
    attention_mask = attention_mask if 0 in attention_mask else None
else:
    attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
        attention_mask=attention_mask,
        input_shape=(input_shape[0], input_shape[-2]),
        inputs_embeds=patch_and_token_embeddings,
        past_key_values_length=past_length,
    )
```

This converts the 2D mask `[batch, seq_len]` into a 4D causal mask for SDPA (Scaled Dot-Product Attention).

---

### 3.3 DTrOCRLMHeadModel.forward - Per-Sample Loss (Lines 184-266)

#### New Parameters

```python
def forward(
    self,
    pixel_values: torch.Tensor,
    input_ids: torch.LongTensor,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = False,
    labels: Optional[torch.LongTensor] = None,
    context_length: Optional[int] = 0,  # NEW
    return_per_sample_loss: Optional[bool] = False,  # NEW
) -> DTrOCRLMHeadModelOutput:
```

#### Forward Through Transformer

```python
transformer_output = self.transformer(
    pixel_values=pixel_values,
    input_ids=input_ids,
    past_key_values=past_key_values,
    position_ids=position_ids,
    attention_mask=attention_mask,
    use_cache=use_cache,
    context_length=context_length  # Pass context_length down
)
logits = self.language_model_head(transformer_output.hidden_states)
```

**Logits shape:** `[batch, seq_len, vocab_size]` where `seq_len = ctx_len + 128 + tgt_len`

#### Loss Computation

**Step 1: Calculate Skip Positions**

```python
# Skip: [context, SEP, IMAGE patches, BOS] to only compute loss on target predictions
skip_positions = context_length + self.image_embedding_length
# context_length: 0 for isolation mode, >0 for contextual mode
# self.image_embedding_length: 128 (number of image patches)
```

**Examples:**
- Contextual mode: context_length=3, skip=3+128=131
- Isolation mode: context_length=0, skip=0+128=128

**Step 2: Shift Logits and Labels**

```python
# Autoregressive: predict token n from tokens < n
shift_logits = logits[..., skip_positions:-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()

batch_size = shift_logits.size(0)
seq_len = shift_logits.size(1)
```

**What's happening:**
- `logits[..., skip_positions:-1, :]`: Take logits from skip_positions to second-last position
- `labels[..., 1:]`: Take labels from second position onwards
- This aligns prediction logits with target labels

**Example (Contextual):**
- Sequence after reordering: [ctx, SEP, img0, ..., img127, BOS, tgt1, tgt2, EOS]
- Logits positions: [0, 1, 2, ..., 130, 131, 132, 133, 134]
- skip_positions = 131
- shift_logits: positions [131:134] → predictions for [BOS, tgt1, tgt2]
- shift_labels: [1:] → [tgt1, tgt2, EOS]
- Alignment: logit[131] predicts tgt1, logit[132] predicts tgt2, logit[133] predicts EOS

**Step 3: Compute Per-Token Loss**

```python
loss_fct = nn.CrossEntropyLoss(reduction="none")
loss_per_token = loss_fct(
    shift_logits.view(-1, shift_logits.size(-1)),  # [batch*seq_len, vocab]
    shift_labels.view(-1)  # [batch*seq_len]
)
# Output shape: [batch*seq_len]
```

**Step 4: Compute Accuracy**

```python
label_matches = shift_labels.view(-1) == torch.argmax(
    torch.nn.functional.softmax(shift_logits.view(-1, shift_logits.size(-1)), dim=-1),
    dim=-1
)
# Output: [batch*seq_len] boolean tensor
```

**Step 5: Reshape for Per-Sample Computation**

```python
loss_2d = loss_per_token.view(batch_size, seq_len)
matches_2d = label_matches.view(batch_size, seq_len)
# Both: [batch, seq_len]
```

**Step 6: Apply Attention Mask**

```python
if attention_mask is not None:
    # Mask corresponds to positions after skip (only target tokens)
    mask = attention_mask[..., 1:].reshape(batch_size, seq_len)

    # Per-sample loss: average over valid tokens
    per_sample_loss = (mask * loss_2d).sum(dim=1) / mask.sum(dim=1)
    # Shape: [batch]

    # Batch-level loss: average across samples
    loss = per_sample_loss.mean()
    # Scalar

    # Batch accuracy
    accuracy = (mask * matches_2d.float()).sum() / mask.sum()
    # Scalar
else:
    # No masking (less common)
    per_sample_loss = loss_2d.mean(dim=1)
    loss = per_sample_loss.mean()
    accuracy = matches_2d.float().mean()
```

**Why masking matters:**
- Padding tokens have mask=0, so they don't contribute to loss
- Only actual target tokens (mask=1) affect training

**Step 7: Return Output**

```python
return DTrOCRLMHeadModelOutput(
    loss=loss,  # Scalar, for backward pass
    logits=logits,  # [batch, full_seq_len, vocab]
    accuracy=accuracy,  # Scalar
    past_key_values=transformer_output.past_key_values,
    per_sample_loss=per_sample_loss if return_per_sample_loss else None
    # [batch] if requested, else None
)
```

---

## 4. dataset.py - Context-Aware Dataset

### New Class: ContextAwareDataset

```python
class ContextAwareDataset(Dataset):
    def __init__(self, images_dir, json_dir, config: DTrOCRConfig, data_frame=None):
        super(ContextAwareDataset, self).__init__()

        self.images_dir = images_dir
        self.json_dir = json_dir
        self.config = config

        if data_frame is None:
            self.df = self._build_dataframe_from_json()
        else:
            self.df = data_frame

        self.processor = DTrOCRProcessor(config, add_eos_token=True, add_bos_token=True)
```

### Dataset Construction from JSON

```python
def _build_dataframe_from_json(self):
    data = []
    json_files = sorted([f for f in os.listdir(self.json_dir) if f.endswith('.json')])

    for json_file in json_files:
        json_path = os.path.join(self.json_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Extract image filename and text
        image_name = os.path.basename(json_data['output_path'])
        text = json_data['text']

        data.append({
            'image_id': image_name,
            'text': text
        })

    df = pd.DataFrame(data)

    # Add previous word as context
    df['prev_text'] = df['text'].shift(1).fillna("")
    # First word has empty context

    return df
```

**Example DataFrame:**

| image_id | text | prev_text |
|----------|------|-----------|
| image_000000.png | আমার | "" |
| image_000001.png | নাম | আমার |
| image_000002.png | রহিম | নাম |

### Data Loading (__getitem__)

```python
def __getitem__(self, index):
    row = self.df.iloc[index]
    image_path = os.path.join(self.images_dir, row['image_id'])
    image = Image.open(image_path).convert('RGB')
    text = str(row['text'])
    prev_text = str(row['prev_text'])

    # Process WITH context (for contextual mode)
    inputs_with_context = self.processor(
        images=image,
        texts=text,
        context_text=prev_text,  # Previous word
        padding=True,
        return_tensors="pt",
        return_labels=True,
    )

    # Process WITHOUT context (for isolation mode)
    inputs_without_context = self.processor(
        images=image,
        texts=text,
        context_text="",  # No context
        padding=True,
        return_tensors="pt",
        return_labels=True,
    )

    return {
        # Contextual mode data
        'pixel_values': inputs_with_context.pixel_values[0],
        'input_ids': inputs_with_context.input_ids[0],
        'attention_mask': inputs_with_context.attention_mask[0],
        'labels': inputs_with_context.labels[0],
        'context_length': inputs_with_context.context_length,

        # Isolation mode data
        'input_ids_isolated': inputs_without_context.input_ids[0],
        'attention_mask_isolated': inputs_without_context.attention_mask[0],
        'labels_isolated': inputs_without_context.labels[0],
        'context_length_isolated': inputs_without_context.context_length,

        # Metadata
        'prev_text': prev_text,
        'text': text
    }
```

**Why Two Versions?**
- Dual-path loss requires both contextual and isolation forward passes
- Dataset provides both versions to avoid reprocessing during training
- pixel_values is shared (same image)

### Train/Test Split

```python
def split_context_data(images_dir, json_dir, config, test_size=0.05, random_seed=42):
    dataset = ContextAwareDataset(images_dir, json_dir, config)
    df = dataset.df

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed)

    train_dataset = ContextAwareDataset(images_dir, json_dir, config,
                                       data_frame=train_df.reset_index(drop=True))
    test_dataset = ContextAwareDataset(images_dir, json_dir, config,
                                      data_frame=test_df.reset_index(drop=True))

    return train_dataset, test_dataset
```

---

## 5. train.py - Dual-Path Training Loop

### Command-Line Arguments

```python
parser = argparse.ArgumentParser(description='Train GraDeT-HTR with optional context-aware mode')
parser.add_argument('--context', action='store_true', help='Enable context-aware training')
parser.add_argument('--images_dir', type=str, default="../sample_train/images")
parser.add_argument('--labels_file', type=str, default="../sample_train/labels/label.csv")
parser.add_argument('--json_dir', type=str, default="../out/single_words/json")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()
```

### Dataset Selection

```python
if args.context:
    print("Using context-aware mode")
    train_dataset, validation_dataset = split_context_data(
        args.images_dir, args.json_dir, config, test_size=0.1
    )
else:
    print("Using standard mode (no context)")
    train_dataset, validation_dataset = split_data(
        args.images_dir, args.labels_file, config
    )
```

### Training Loop - Context Mode

```python
for epoch in range(EPOCHS):
    model.train()
    epoch_losses, epoch_ctx_losses, epoch_iso_losses = [], [], []
    epoch_ctx_accs, epoch_iso_accs = [], []

    for inputs in tqdm.tqdm(train_dataloader, desc=f'Epoch {epoch + 1}'):
        optimizer.zero_grad()

        pixel_values = inputs['pixel_values'].to(device)
        labels = inputs['labels'].to(device)

        if args.context:
            # === CONTEXTUAL FORWARD PASS ===

            # Prepare contextual inputs
            input_ids_ctx = inputs['input_ids'].to(device)
            attention_mask_ctx = inputs['attention_mask'].to(device)
            context_length = inputs['context_length'][0].item() \
                if isinstance(inputs['context_length'], torch.Tensor) \
                else inputs['context_length']

            # Forward pass with context
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                outputs_ctx = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids_ctx,
                    attention_mask=attention_mask_ctx,
                    labels=labels,
                    context_length=context_length,
                    return_per_sample_loss=True  # Request per-sample losses
                )
                loss_ctx = outputs_ctx.loss  # Scalar
                per_sample_loss_ctx = outputs_ctx.per_sample_loss  # [batch]
                acc_ctx = outputs_ctx.accuracy  # Scalar
```

**Key Points:**
- `context_length` extracted from batch (may be tensor or int)
- `return_per_sample_loss=True` enables difficulty weighting
- Mixed precision (`torch.autocast`) for speed

```python
            # === ISOLATION FORWARD PASS ===

            # Prepare isolation inputs (no context)
            input_ids_iso = inputs['input_ids_isolated'].to(device)
            attention_mask_iso = inputs['attention_mask_isolated'].to(device)
            context_length_iso = inputs['context_length_isolated'][0].item() \
                if isinstance(inputs['context_length_isolated'], torch.Tensor) \
                else inputs['context_length_isolated']
            labels_iso = inputs['labels_isolated'].to(device)

            # Forward pass without context
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                outputs_iso = model(
                    pixel_values=pixel_values,  # Same image!
                    input_ids=input_ids_iso,
                    attention_mask=attention_mask_iso,
                    labels=labels_iso,
                    context_length=context_length_iso,  # Should be 0
                    return_per_sample_loss=True
                )
                loss_iso = outputs_iso.loss  # Scalar
                per_sample_loss_iso = outputs_iso.per_sample_loss  # [batch]
                acc_iso = outputs_iso.accuracy  # Scalar
```

**Important:**
- Same `pixel_values` (image) used in both passes
- Different `input_ids` (with/without context)
- `context_length_iso` should be 0

```python
            # === DIFFICULTY-AWARE WEIGHTING ===

            # Compute weights based on isolation difficulty
            batch_size = per_sample_loss_iso.size(0)
            weights = (per_sample_loss_iso / per_sample_loss_iso.sum()) * batch_size
            # Normalize to sum to batch_size

            # Clamp weights to prevent extreme values
            weights = torch.clamp(weights, min=0.1, max=10.0)
```

**Weight Computation Example:**
```
Batch of 4 samples:
per_sample_loss_iso = [2.0, 5.0, 1.0, 4.0]
sum = 12.0

Raw weights:
w = [2.0/12.0, 5.0/12.0, 1.0/12.0, 4.0/12.0] * 4
  = [0.667, 1.667, 0.333, 1.333]

After clamping [0.1, 10.0]:
w = [0.667, 1.667, 0.333, 1.333] (no change, within range)

Interpretation:
- Sample 2 (loss=5.0) gets highest weight (1.667) → hardest
- Sample 3 (loss=1.0) gets lowest weight (0.333) → easiest
```

```python
            # Combined per-sample loss (sum of contextual and isolation)
            combined_per_sample_loss = per_sample_loss_ctx + per_sample_loss_iso
            # [batch]

            # Apply difficulty weighting
            weighted_loss = (weights * combined_per_sample_loss).mean()
            # Scalar: weighted average of combined losses
```

**Mathematical Expression:**
```
J = (1/B) * Σ_{i=1}^{B} w_i * (L_ctx_i + L_iso_i)

Where:
- B = batch_size
- w_i = difficulty weight for sample i
- L_ctx_i = contextual loss for sample i
- L_iso_i = isolation loss for sample i
```

```python
            # === BACKWARD PASS ===

            scaler.scale(weighted_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track metrics
            epoch_losses.append(weighted_loss.item())
            epoch_ctx_losses.append(loss_ctx.item())
            epoch_iso_losses.append(loss_iso.item())
            epoch_ctx_accs.append(acc_ctx.item())
            epoch_iso_accs.append(acc_iso.item())
            epoch_accuracies.append((acc_ctx.item() + acc_iso.item()) / 2)
```

### Validation Loop

```python
def evaluate_context_model(model, dataloader, device, use_amp=True):
    model.eval()
    val_losses, val_ctx_losses, val_iso_losses = [], [], []
    val_accs = []

    with torch.no_grad():
        for inputs in dataloader:
            pixel_values = inputs['pixel_values'].to(device)
            labels = inputs['labels'].to(device)

            # Contextual
            input_ids_ctx = inputs['input_ids'].to(device)
            attention_mask_ctx = inputs['attention_mask'].to(device)
            context_length = inputs['context_length'][0].item() \
                if isinstance(inputs['context_length'], torch.Tensor) \
                else inputs['context_length']

            # Isolation
            input_ids_iso = inputs['input_ids_isolated'].to(device)
            attention_mask_iso = inputs['attention_mask_isolated'].to(device)
            labels_iso = inputs['labels_isolated'].to(device)
            context_length_iso = inputs['context_length_isolated'][0].item() \
                if isinstance(inputs['context_length_isolated'], torch.Tensor) \
                else inputs['context_length_isolated']

            # Forward passes (no per-sample loss needed for validation)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                outputs_ctx = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids_ctx,
                    attention_mask=attention_mask_ctx,
                    labels=labels,
                    context_length=context_length
                )
                outputs_iso = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids_iso,
                    attention_mask=attention_mask_iso,
                    labels=labels_iso,
                    context_length=context_length_iso
                )

            # Compute metrics
            loss_net = outputs_ctx.loss + outputs_iso.loss
            val_losses.append(loss_net.item())
            val_ctx_losses.append(outputs_ctx.loss.item())
            val_iso_losses.append(outputs_iso.loss.item())
            val_accs.append((outputs_ctx.accuracy.item() + outputs_iso.accuracy.item()) / 2)

    return (
        sum(val_losses) / len(val_losses),
        sum(val_accs) / len(val_accs),
        sum(val_ctx_losses) / len(val_ctx_losses),
        sum(val_iso_losses) / len(val_iso_losses)
    )
```

### Training Output

```python
print(f"\nEpoch {epoch + 1}/{EPOCHS}:")
print(f"  Train - Loss: {train_losses[-1]:.4f} | Acc: {train_accuracies[-1]:.4f}")
print(f"    Ctx Loss: {sum(epoch_ctx_losses)/len(epoch_ctx_losses):.4f} | " \
      f"Iso Loss: {sum(epoch_iso_losses)/len(epoch_iso_losses):.4f}")
print(f"    Ctx Acc: {sum(epoch_ctx_accs)/len(epoch_ctx_accs):.4f} | " \
      f"Iso Acc: {sum(epoch_iso_accs)/len(epoch_iso_accs):.4f}")
print(f"  Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
print(f"    Ctx Loss: {val_ctx_loss:.4f} | Iso Loss: {val_iso_loss:.4f}")
```

**Example Output:**
```
Epoch 1/10:
  Train - Loss: 2.3456 | Acc: 0.7234
    Ctx Loss: 1.8234 | Iso Loss: 2.8678
    Ctx Acc: 0.7891 | Iso Acc: 0.6577
  Val   - Loss: 2.4567 | Acc: 0.7123
    Ctx Loss: 1.9345 | Iso Loss: 2.9789
```

---

## 6. Complete Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT DATA                                  │
│  Image: word_image.png                                            │
│  Text: "বাংলা" (current word)                                     │
│  Context: "আমার" (previous word)                                  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    PROCESSOR.PY                                   │
│  1. Tokenize context: "আমার" → [15, 23]                          │
│  2. Add SEP: [15, 23, 3]                                          │
│  3. Tokenize text: "বাংলা" → [1, 45, 67, 2]                      │
│  4. Concatenate: [15, 23, 3, 1, 45, 67, 2]                        │
│  5. context_length = 3                                            │
│  6. Process image → pixel_values                                  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                  MODEL.PY - DTrOCRModel                           │
│  1. Embed tokens: [15, 23, 3, 1, 45, 67, 2] → token_embeddings   │
│  2. Embed image: pixel_values → patch_embeddings (128 patches)   │
│  3. Split tokens:                                                 │
│     - context_embeddings = [:3]  → [15, 23, 3]                   │
│     - target_embeddings = [3:]   → [1, 45, 67, 2]                │
│  4. Reorder:                                                      │
│     [context_embeddings, patch_embeddings, target_embeddings]    │
│     = [15, 23, 3, img0, ..., img127, 1, 45, 67, 2]              │
│  5. Adjust attention mask:                                        │
│     [1, 1, 1, 1, ..., 1, 1, 1, 1, 1] (3+128+4 = 135)            │
│  6. Pass through transformer blocks                               │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              MODEL.PY - DTrOCRLMHeadModel                         │
│  1. Get logits: [batch, 135, vocab_size]                         │
│  2. skip_positions = 3 + 128 = 131                               │
│  3. shift_logits = logits[131:134]  (BOS, tgt1, tgt2)           │
│  4. shift_labels = labels[1:4]      (tgt1, tgt2, EOS)           │
│  5. Compute per-token loss                                        │
│  6. Apply attention mask                                          │
│  7. Compute per-sample loss: [batch]                             │
│  8. Return loss, accuracy, per_sample_loss                       │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    TRAIN.PY - Training Loop                       │
│  1. Forward pass (contextual): L_ctx, per_sample_loss_ctx        │
│  2. Forward pass (isolation): L_iso, per_sample_loss_iso         │
│  3. Compute difficulty weights:                                   │
│     w_i = (L_iso_i / Σ L_iso) * batch_size                       │
│  4. Clamp weights: [0.1, 10.0]                                   │
│  5. Combined loss: (L_ctx + L_iso) per sample                    │
│  6. Weighted loss: mean(w_i * combined_i)                        │
│  7. Backward pass                                                 │
│  8. Update weights                                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## Summary of Key Implementation Details

### 1. Sequence Structure
- **Standard**: `[IMAGE (128), BOS, target]`
- **Context**: `[context, SEP, IMAGE (128), BOS, target]`

### 2. Loss Computation
- **Skip positions**: context_length + 128 for contextual, 128 for isolation
- **Per-sample**: Enables difficulty weighting
- **Dual-path**: L_net = L_ctx + L_iso

### 3. Attention Mask
- Adjusted to match reordered sequence
- Handles context, image patches, and target tokens
- Proper masking for padding tokens

### 4. Difficulty Weighting
- Based on isolation loss (visual difficulty)
- Normalized and clamped [0.1, 10.0]
- Applied to combined per-sample loss

### 5. Training Efficiency
- 2x forward passes per batch
- Mixed precision (FP16) for speed
- Batch size typically reduced (16 vs 32)

---

## File Change Summary

| File | Lines Changed | Key Modifications |
|------|---------------|-------------------|
| data.py | +2 | Added `per_sample_loss` and `context_length` fields |
| processor.py | ~70 | Context tokenization and concatenation logic |
| model.py | ~150 | Embedding reordering, attention mask adjustment, per-sample loss |
| dataset.py | ~100 | ContextAwareDataset class with dual processing |
| train.py | ~170 | Dual-path training loop with difficulty weighting |

**Total**: ~490 lines of new/modified code

---
