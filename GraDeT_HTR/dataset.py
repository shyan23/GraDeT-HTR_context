import os
import json
import pandas as pd
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


class ContextAwareDataset(Dataset):
    """
    Dataset for context-aware handwritten text recognition.
    Each sample includes the current word image and the previous word's text as context.
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
        """Build a dataframe from JSON files containing text and image paths."""
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

        # Add previous word context
        df['prev_text'] = df['text'].shift(1).fillna("")  # First word has empty context

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = os.path.join(self.images_dir, row['image_id'])
        image = Image.open(image_path).convert('RGB')
        text = str(row['text'])
        prev_text = str(row['prev_text'])

        # Process image ONCE — pixel_values are identical for both paths
        pixel_values = self.processor.vit_processor(
            image, input_data_format='channels_last', return_tensors="pt"
        )["pixel_values"][0]

        # Tokenize with context (contextual path) — text only, no image
        inputs_with_context = self.processor(
            images=None,
            texts=text,
            context_text=prev_text,
            padding=True,
            return_tensors="pt",
            return_labels=True,
        )

        # Tokenize without context (isolation path) — text only, no image
        inputs_without_context = self.processor(
            images=None,
            texts=text,
            context_text="",
            padding=True,
            return_tensors="pt",
            return_labels=True,
        )

        return {
            # Shared image (processed once)
            'pixel_values': pixel_values,

            # Contextual mode inputs
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
    """Split context-aware dataset into train and test sets."""
    dataset = ContextAwareDataset(images_dir, json_dir, config)
    df = dataset.df

    # Split into train + validation/test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed)

    train_dataset = ContextAwareDataset(images_dir, json_dir, config, data_frame=train_df.reset_index(drop=True))
    test_dataset = ContextAwareDataset(images_dir, json_dir, config, data_frame=test_df.reset_index(drop=True))

    return train_dataset, test_dataset
