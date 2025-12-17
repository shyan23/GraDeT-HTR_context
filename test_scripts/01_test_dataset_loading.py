#!/usr/bin/env python3
"""
Test 1: Dataset Loading Validation
Tests that datasets load correctly for both standard and context-aware modes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GraDeT_HTR'))

from dataset import split_data, split_context_data
from config import DTrOCRConfig

def test_standard_dataset():
    """Test standard dataset loading (CSV-based)"""
    print("=" * 60)
    print("TEST 1A: Standard Dataset Loading (CSV)")
    print("=" * 60)

    try:
        config = DTrOCRConfig()
        images_dir = "../sample_train/images"
        labels_file = "../sample_train/labels/label.csv"

        print(f"Loading from: {images_dir}")
        print(f"Labels: {labels_file}")

        train_dataset, test_dataset = split_data(images_dir, labels_file, config, test_size=0.1)

        print(f"✅ Train size: {len(train_dataset)}")
        print(f"✅ Test size: {len(test_dataset)}")

        # Test loading one sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\n📦 Sample keys: {sample.keys()}")
            print(f"   - pixel_values shape: {sample['pixel_values'].shape}")
            print(f"   - input_ids shape: {sample['input_ids'].shape}")
            print(f"   - attention_mask shape: {sample['attention_mask'].shape}")
            print(f"   - labels shape: {sample['labels'].shape}")
            print("\n✅ Standard dataset loading PASSED\n")
            return True
        else:
            print("❌ Dataset is empty!")
            return False

    except Exception as e:
        print(f"❌ Standard dataset loading FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_dataset():
    """Test context-aware dataset loading (JSON-based)"""
    print("=" * 60)
    print("TEST 1B: Context-Aware Dataset Loading (JSON)")
    print("=" * 60)

    try:
        config = DTrOCRConfig()
        images_dir = "../out/single_words/images"
        json_dir = "../out/single_words/json"

        print(f"Loading from: {images_dir}")
        print(f"JSON dir: {json_dir}")

        # Check if directories exist
        if not os.path.exists(images_dir):
            print(f"❌ Images directory not found: {images_dir}")
            return False
        if not os.path.exists(json_dir):
            print(f"❌ JSON directory not found: {json_dir}")
            return False

        train_dataset, test_dataset = split_context_data(images_dir, json_dir, config, test_size=0.1)

        print(f"✅ Train size: {len(train_dataset)}")
        print(f"✅ Test size: {len(test_dataset)}")

        # Test loading one sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\n📦 Sample keys: {sample.keys()}")
            print(f"   - pixel_values shape: {sample['pixel_values'].shape}")
            print(f"   - input_ids (ctx) shape: {sample['input_ids'].shape}")
            print(f"   - input_ids (iso) shape: {sample['input_ids_isolated'].shape}")
            print(f"   - context_length: {sample['context_length']}")
            print(f"   - prev_text: '{sample['prev_text']}'")
            print(f"   - text: '{sample['text']}'")

            # Validate dual-path data
            assert 'input_ids_isolated' in sample, "Missing isolation mode data"
            assert 'context_length' in sample, "Missing context_length"

            print("\n✅ Context-aware dataset loading PASSED\n")
            return True
        else:
            print("❌ Dataset is empty!")
            return False

    except Exception as e:
        print(f"❌ Context-aware dataset loading FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "🧪 " * 20)
    print("DATASET LOADING TESTS")
    print("🧪 " * 20 + "\n")

    # Change to GraDeT_HTR directory
    os.chdir(os.path.join(os.path.dirname(__file__), '..', 'GraDeT_HTR'))

    results = []
    results.append(("Standard Dataset", test_standard_dataset()))
    results.append(("Context Dataset", test_context_dataset()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("🎉 ALL TESTS PASSED!" if all_passed else "⚠️  SOME TESTS FAILED"))

    sys.exit(0 if all_passed else 1)
