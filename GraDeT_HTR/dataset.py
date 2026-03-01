import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image

from config import DTrOCRConfig
from processor import DTrOCRProcessor

class HandwrittenDataset(Dataset):
    def __init__(self, images_dir, data_frame, config: DTrOCRConfig):
        super(HandwrittenDataset, self).__init__()

        self.images_dir = images_dir
        self.df = data_frame

        self.image_ids = self.df["image_id"].values
        self.texts = self.df["text"].values

        self.processor = DTrOCRProcessor(config, add_eos_token=True, add_bos_token=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.images_dir, str(self.image_ids[index]))).convert('RGB')
        text = str(self.texts[index])

        inputs = self.processor(
            images=image,
            texts=text,
            padding=True,
            return_tensors="pt",
            return_labels=True,
        )

        return {
            'pixel_values': inputs.pixel_values[0],
            'input_ids': inputs.input_ids[0],
            'attention_mask': inputs.attention_mask[0],
            'labels': inputs.labels[0]
        }


def _rebuild_prev_text(df):
    """Build prev_text column respecting line_id boundaries.

    For each row, prev_text is the text of the previous row IF they share
    the same line_id. At sentence boundaries (line_id changes) or for
    the first row, prev_text is empty string.

    If line_id is missing or all -1, falls back to blind shift(1) for
    backward compatibility with old data that has no line_id.
    """
    if 'line_id' not in df.columns or (df['line_id'] == -1).all():
        # Backward compat: no line_id info, use blind shift
        df['prev_text'] = df['text'].shift(1).fillna("")
        return df

    prev_texts = []
    for i in range(len(df)):
        if i == 0:
            prev_texts.append("")
        elif df.iloc[i]['line_id'] == df.iloc[i - 1]['line_id']:
            prev_texts.append(str(df.iloc[i - 1]['text']))
        else:
            # Sentence boundary: first word of new sentence
            prev_texts.append("")
    df['prev_text'] = prev_texts
    return df


class ContextAwareDataset(Dataset):
    """
    Dataset for context-aware handwritten text recognition.
    Each sample includes the current word image and the previous word's text as context.
    Respects sentence boundaries via line_id — no cross-sentence context bleeding.
    """
    def __init__(self, images_dir, json_dir, config: DTrOCRConfig, data_frame=None):
        super(ContextAwareDataset, self).__init__()

        self.images_dir = images_dir
        self.json_dir = json_dir
        self.config = config

        # If dataframe is not provided, build it from JSON files
        if data_frame is None:
            self.df = self._build_dataframe_from_json()
        else:
            self.df = data_frame

        self.processor = DTrOCRProcessor(config, add_eos_token=True, add_bos_token=True)

    def _build_dataframe_from_json(self):
        """Build a dataframe from JSON files containing text, image paths, and line_id."""
        data = []
        json_files = sorted([f for f in os.listdir(self.json_dir) if f.endswith('.json')])

        for json_file in json_files:
            json_path = os.path.join(self.json_dir, json_file)
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Extract image filename and text
            image_name = os.path.basename(json_data['output_path'])
            text = json_data['text']
            line_id = json_data.get('line_id', -1)
            word_index = json_data.get('word_index', -1)

            data.append({
                'image_id': image_name,
                'text': text,
                'line_id': line_id,
                'word_index': word_index,
            })

        df = pd.DataFrame(data)

        # Build prev_text respecting sentence boundaries
        df = _rebuild_prev_text(df)

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = os.path.join(self.images_dir, row['image_id'])
        image = Image.open(image_path).convert('RGB')
        text = str(row['text'])
        prev_text = str(row['prev_text'])

        # Process current word with context
        inputs_with_context = self.processor(
            images=image,
            texts=text,
            context_text=prev_text,  # Previous word as context
            padding=True,
            return_tensors="pt",
            return_labels=True,
        )

        # Process current word without context (for isolation mode)
        inputs_without_context = self.processor(
            images=image,
            texts=text,
            context_text="",  # No context
            padding=True,
            return_tensors="pt",
            return_labels=True,
        )

        return {
            # Contextual mode inputs
            'pixel_values': inputs_with_context.pixel_values[0],
            'input_ids': inputs_with_context.input_ids[0],
            'attention_mask': inputs_with_context.attention_mask[0],
            'labels': inputs_with_context.labels[0],
            'context_length': inputs_with_context.context_length,

            # Isolation mode inputs
            'input_ids_isolated': inputs_without_context.input_ids[0],
            'attention_mask_isolated': inputs_without_context.attention_mask[0],
            'labels_isolated': inputs_without_context.labels[0],
            'context_length_isolated': inputs_without_context.context_length,

            # Metadata
            'prev_text': prev_text,
            'text': text
        }


def split_data(images_dir, labels_file, config, test_size=0.05, random_seed=42):
    df = pd.read_csv(labels_file)
    # Split into train + validation/test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed)
    train_dataset = HandwrittenDataset(images_dir, train_df, config)
    test_dataset = HandwrittenDataset(images_dir, test_df, config)

    return train_dataset, test_dataset


def split_context_data(images_dir, json_dir, config, test_size=0.05, random_seed=42):
    """Split context-aware dataset into train and test sets.

    Splits by sentence (line_id) groups so that all words from the same
    sentence stay in the same split. This prevents broken context at
    split boundaries.
    """
    dataset = ContextAwareDataset(images_dir, json_dir, config)
    df = dataset.df

    if 'line_id' not in df.columns or (df['line_id'] == -1).all():
        # No line_id: fall back to row-level split (backward compat)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed)
    else:
        # Split by unique line_id values, keeping whole sentences together
        unique_line_ids = df['line_id'].unique()
        train_line_ids, test_line_ids = train_test_split(
            unique_line_ids, test_size=test_size, random_state=random_seed
        )
        train_df = df[df['line_id'].isin(train_line_ids)]
        test_df = df[df['line_id'].isin(test_line_ids)]

    # Rebuild prev_text after splitting (boundaries may have changed)
    train_df = _rebuild_prev_text(train_df.reset_index(drop=True))
    test_df = _rebuild_prev_text(test_df.reset_index(drop=True))

    train_dataset = ContextAwareDataset(images_dir, json_dir, config, data_frame=train_df)
    test_dataset = ContextAwareDataset(images_dir, json_dir, config, data_frame=test_df)

    return train_dataset, test_dataset
