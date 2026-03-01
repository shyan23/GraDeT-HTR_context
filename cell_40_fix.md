
# Fetch the RoPE branch and checkout its architecture files
# Adds: rope.py, rope_attention.py
# Patches: model.py (RoPE support), config.py (use_rope flag),
#          dataset.py (process image once), utils.py (strict param)
!cd /content/GraDeT-HTR && git fetch origin RoPE
!cd /content/GraDeT-HTR && git checkout origin/RoPE -- \
    GraDeT_HTR/rope.py \
    GraDeT_HTR/rope_attention.py \
    GraDeT_HTR/model.py \
    GraDeT_HTR/config.py \
    GraDeT_HTR/dataset.py \
    GraDeT_HTR/utils.py

print("RoPE files applied. Verifying...")
import os
for f in ['rope.py', 'rope_attention.py']:
    path = os.path.join(REPO_ROOT, 'GraDeT_HTR', f)
    print(f"  {f}: {'OK' if os.path.exists(path) else 'MISSING'}")

# --- Fix processor.py: empty context must still prepend padding block ---
# Without this, first-word-of-sentence samples (empty prev_text) get input_ids
# of length 128, while samples with context get 148 -> collation crash.
# We overwrite processor.py with the corrected version directly.
proc_path = os.path.join(REPO_ROOT, 'GraDeT_HTR', 'processor.py')
with open(proc_path, 'w') as f:
    f.write('''from transformers import GPT2Tokenizer, AutoImageProcessor
from bntokenizer import BnGraphemizerProcessor

from PIL import Image
from typing import List, Union
from config import DTrOCRConfig
from data import DTrOCRProcessorOutput


class DTrOCRProcessor:
    def __init__(self, config: DTrOCRConfig, add_bos_token: bool = False, add_eos_token: bool = False):
        self.vit_processor = AutoImageProcessor.from_pretrained(
            config.vit_hf_model,
            size={
                "height": config.image_size[0],
                "width": config.image_size[1]
            },
            use_fast=True
        )
        self.max_context_length = getattr(config, "max_context_length", 20)
        self.tokeniser = BnGraphemizerProcessor(
            config.bn_vocab_file,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            model_max_length=config.max_position_embeddings - int(
                (config.image_size[0] / config.patch_size[0]) * (config.image_size[1] / config.patch_size[1])
            )
        )

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]] = None,
        texts: Union[str, List[str]] = None,
        context_text: Union[str, List[str]] = None,
        return_labels: bool = False,
        input_data_format: str = "channels_last",
        padding: Union[bool, str] = False,
        *args,
        **kwargs
    ) -> DTrOCRProcessorOutput:
        import torch

        context_length = 0

        # Process target text
        text_inputs = self.tokeniser(
            texts, padding=padding
        ) if texts is not None else None

        # If context_text is provided (even empty), prepend a fixed-length
        # context block so all samples have identical sequence length for batching
        if context_text is not None and text_inputs is not None:
            pad_token_id = self.tokeniser.pad_token_id

            if context_text != "":
                context_inputs = self.tokeniser(context_text, padding=False)
                sep_token_id = self.tokeniser.eos_token_id
                context_ids = context_inputs["input_ids"]

                if isinstance(context_ids, list):
                    if len(context_ids) > 0 and isinstance(context_ids[0], list):
                        context_ids = context_ids[0]
                elif hasattr(context_ids, "tolist"):
                    context_ids = context_ids.tolist()
                    if isinstance(context_ids[0], list):
                        context_ids = context_ids[0]

                context_ids = context_ids[:self.max_context_length - 1]
                context_with_sep = context_ids + [sep_token_id]

                actual_len = len(context_with_sep)
                pad_needed = self.max_context_length - actual_len
                context_padded = context_with_sep + [pad_token_id] * pad_needed
                context_attn_mask = [1] * actual_len + [0] * pad_needed
            else:
                # Empty context: all padding, fully masked out
                context_padded = [pad_token_id] * self.max_context_length
                context_attn_mask = [0] * self.max_context_length

            context_length = self.max_context_length

            target_ids = text_inputs["input_ids"]
            if hasattr(target_ids, "tolist"):
                target_ids = target_ids.tolist()
                if isinstance(target_ids[0], list):
                    target_ids = target_ids[0]
            elif isinstance(target_ids, list) and len(target_ids) > 0 and isinstance(target_ids[0], list):
                target_ids = target_ids[0]

            combined_ids = context_padded + target_ids
            text_inputs["input_ids"] = torch.tensor([combined_ids])

            target_mask = text_inputs["attention_mask"]
            if hasattr(target_mask, "tolist"):
                target_mask = target_mask.tolist()
                if isinstance(target_mask[0], list):
                    target_mask = target_mask[0]
            elif isinstance(target_mask, list) and len(target_mask) > 0 and isinstance(target_mask[0], list):
                target_mask = target_mask[0]

            combined_mask = context_attn_mask + target_mask
            text_inputs["attention_mask"] = torch.tensor([combined_mask])

        image_inputs = self.vit_processor(
            images, input_data_format=input_data_format, *args, **kwargs
        ) if images is not None else None

        return DTrOCRProcessorOutput(
            pixel_values=image_inputs["pixel_values"] if images is not None else None,
            input_ids=text_inputs["input_ids"] if texts is not None else None,
            attention_mask=text_inputs["attention_mask"] if texts is not None else None,
            labels=text_inputs["input_ids"] if texts is not None and return_labels else None,
            context_length=context_length
        )
''')
print("processor.py overwritten with fixed version (empty context gets padding block)")

