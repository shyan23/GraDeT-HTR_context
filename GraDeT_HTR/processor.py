from transformers import GPT2Tokenizer, AutoImageProcessor
from bntokenizer import BnGraphemizerProcessor

from PIL import Image
from typing import List, Union
from config import DTrOCRConfig
from data import DTrOCRProcessorOutput


class DTrOCRProcessor:
    def __init__(self, config: DTrOCRConfig, add_bos_token: bool = False, add_eos_token: bool = False):
        
        """
        AutoImageProcessor.from_pretrained(...) -> 
        Resizing
        Normalization
        Channel ordering
        Converting PIL images → tensors
        So ViT gets a perfect pixel_values tensor (batch, channels, H, W).
        """
        
        
        
        self.vit_processor = AutoImageProcessor.from_pretrained(
            config.vit_hf_model,
            size={
                "height": config.image_size[0],
                'width': config.image_size[1]
            },
            use_fast=True
        )
        """
        This tokenizer:

        Loads custom grapheme vocab (Bangla)
        Converts text → token sequences
        Creates attention masks
        Optionally adds BOS/EOS
        Ensures text length fits in the decoder
        """
        self.max_context_length = getattr(config, 'max_context_length', 20)
        self.tokeniser = BnGraphemizerProcessor(
            config.bn_vocab_file,
            add_bos_token=add_bos_token,
            add_eos_token = add_eos_token,
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
        input_data_format: str = 'channels_last',
        padding: Union[bool, str] = False,
        *args,
        **kwargs
    ) -> DTrOCRProcessorOutput:
        """
        Process images and texts with optional context.

        Returns input_ids in format: [context_tokens, SEP, target_tokens]
        Model will reorder to: [context, SEP, IMAGE_PATCHES, BOS, target]

        context_text: Previous word text to be prepended before image patches
        """
        import torch

        context_length = 0

        # Process target text
        text_inputs = self.tokeniser(
            texts, padding=padding
        ) if texts is not None else None

        # If context is provided, prepend it with separator and pad to fixed length
        if context_text is not None and context_text != "" and text_inputs is not None:
            # Tokenize context (no padding — we pad to fixed length ourselves)
            context_inputs = self.tokeniser(
                context_text, padding=False
            )

            sep_token_id = self.tokeniser.eos_token_id  # Reuse EOS as separator
            pad_token_id = self.tokeniser.pad_token_id
            context_ids = context_inputs['input_ids']

            # Handle list/tensor conversion
            if isinstance(context_ids, list):
                if len(context_ids) > 0 and isinstance(context_ids[0], list):
                    context_ids = context_ids[0]
            elif hasattr(context_ids, 'tolist'):
                context_ids = context_ids.tolist()
                if isinstance(context_ids[0], list):
                    context_ids = context_ids[0]

            # Truncate context to leave room for SEP within max_context_length
            context_ids = context_ids[:self.max_context_length - 1]

            # Create context with separator: [context_tokens] + [SEP]
            context_with_sep = context_ids + [sep_token_id]

            # Pad to fixed max_context_length so all samples have identical shape
            actual_len = len(context_with_sep)
            pad_needed = self.max_context_length - actual_len
            context_padded = context_with_sep + [pad_token_id] * pad_needed
            context_attn_mask = [1] * actual_len + [0] * pad_needed

            # Fixed context_length for ALL samples (enables uniform batching)
            context_length = self.max_context_length

            # Get target tokens
            target_ids = text_inputs['input_ids']
            if hasattr(target_ids, 'tolist'):
                target_ids = target_ids.tolist()
                if isinstance(target_ids[0], list):
                    target_ids = target_ids[0]
            elif isinstance(target_ids, list) and len(target_ids) > 0 and isinstance(target_ids[0], list):
                target_ids = target_ids[0]

            # Concatenate: [context_padded] + [target_tokens]
            combined_ids = context_padded + target_ids
            text_inputs['input_ids'] = torch.tensor([combined_ids])

            # Update attention mask
            target_mask = text_inputs['attention_mask']
            if hasattr(target_mask, 'tolist'):
                target_mask = target_mask.tolist()
                if isinstance(target_mask[0], list):
                    target_mask = target_mask[0]
            elif isinstance(target_mask, list) and len(target_mask) > 0 and isinstance(target_mask[0], list):
                target_mask = target_mask[0]

            combined_mask = context_attn_mask + target_mask
            text_inputs['attention_mask'] = torch.tensor([combined_mask])

        # Process images
        image_inputs = self.vit_processor(
            images, input_data_format=input_data_format, *args, **kwargs
        ) if images is not None else None

        return DTrOCRProcessorOutput(
            pixel_values=image_inputs["pixel_values"] if images is not None else None,
            input_ids=text_inputs['input_ids'] if texts is not None else None,
            attention_mask=text_inputs['attention_mask'] if texts is not None else None,
            labels=text_inputs['input_ids'] if texts is not None and return_labels else None,
            context_length=context_length
        )
