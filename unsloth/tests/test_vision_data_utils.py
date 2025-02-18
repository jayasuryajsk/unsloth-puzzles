import unittest
import torch
from PIL import Image
import numpy as np
import io
import warnings
from transformers import AutoProcessor
from ..models.vision_data_utils import UnslothVisionDataCollator, ImageResizeTransform

class MockProcessor:
    """Mock processor for testing without loading real models"""
    def __init__(self):
        self.image_processor = type('obj', (object,), {'size': 224})
        
    def __call__(self, images=None, return_tensors="pt", **kwargs):
        if images is None:
            return None
        # Convert all images to tensors of expected shape
        pixel_values = torch.stack([
            torch.tensor(np.array(img), dtype=torch.float32) 
            for img in images
        ])
        return type('obj', (object,), {'pixel_values': pixel_values})

def create_dummy_image(size=(100, 100)):
    """Create a dummy PIL Image for testing"""
    return Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))

def create_dummy_text_example():
    """Create a dummy text-only example"""
    return {
        "input_ids": [1, 2, 3, 4, 5],
        "attention_mask": [1, 1, 1, 1, 1],
        "labels": [1, 2, 3, 4, 5]
    }

def create_dummy_image_example(num_images=1):
    """Create a dummy example with image(s)"""
    example = create_dummy_text_example()
    if num_images == 1:
        example["images"] = create_dummy_image()
    else:
        example["images"] = [create_dummy_image() for _ in range(num_images)]
    return example

class TestUnslothVisionDataCollator(unittest.TestCase):
    def setUp(self):
        self.processor = MockProcessor()
        self.collator = UnslothVisionDataCollator(
            processor=self.processor,
            max_seq_length=10,
            max_image_size=224
        )

    def test_text_only_batch(self):
        """Test processing a batch with only text examples"""
        batch = [create_dummy_text_example() for _ in range(3)]
        output = self.collator(batch)
        
        self.assertIn("input_ids", output)
        self.assertIn("attention_mask", output)
        self.assertIn("labels", output)
        self.assertNotIn("pixel_values", output)
        self.assertEqual(output["input_ids"].shape[0], 3)

    def test_single_image_batch(self):
        """Test processing a batch where each example has one image"""
        batch = [create_dummy_image_example() for _ in range(3)]
        output = self.collator(batch)
        
        self.assertIn("pixel_values", output)
        self.assertEqual(len(output["image_indices"]), 3)
        self.assertEqual(output["pixel_values"].shape[0], 3)

    def test_multi_image_example(self):
        """Test processing examples with multiple images"""
        batch = [
            create_dummy_image_example(num_images=2),
            create_dummy_image_example(num_images=1)
        ]
        output = self.collator(batch)
        
        self.assertIn("pixel_values", output)
        self.assertEqual(output["pixel_values"].shape[0], 3)  # Total 3 images
        self.assertEqual(len(output["image_indices"]), 2)     # 2 examples

    def test_mixed_batch(self):
        """Test processing a batch with both text-only and image examples"""
        batch = [
            create_dummy_text_example(),
            create_dummy_image_example(),
            create_dummy_text_example()
        ]
        output = self.collator(batch)
        
        self.assertIn("pixel_values", output)
        self.assertEqual(len(output["image_indices"]), 3)
        self.assertTrue(output["image_indices"][0] is None)  # Text-only
        self.assertTrue(output["image_indices"][2] is None)  # Text-only

    def test_sequence_truncation(self):
        """Test that sequences longer than max_seq_length are truncated"""
        long_example = create_dummy_text_example()
        long_example["input_ids"] = list(range(20))  # Longer than max_seq_length
        long_example["attention_mask"] = [1] * 20
        
        output = self.collator([long_example])
        self.assertEqual(output["input_ids"].shape[1], 10)  # Should be truncated

    def test_padding(self):
        """Test padding of sequences with different lengths"""
        batch = [
            {"input_ids": [1, 2], "attention_mask": [1, 1]},
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        ]
        output = self.collator(batch)
        
        self.assertEqual(output["input_ids"].shape, (2, 3))
        self.assertEqual(output["attention_mask"].shape, (2, 3))

    def test_image_resizing(self):
        """Test that images are resized correctly"""
        large_image = create_dummy_image(size=(300, 300))
        batch = [{"input_ids": [1], "attention_mask": [1], "images": large_image}]
        output = self.collator(batch)
        
        # Check that image was resized to max_image_size
        self.assertEqual(output["pixel_values"].shape[-2:], (224, 224))

    def test_invalid_image(self):
        """Test handling of invalid/corrupted images"""
        invalid_example = create_dummy_text_example()
        invalid_example["images"] = "nonexistent.jpg"
        
        with warnings.catch_warnings(record=True) as w:
            output = self.collator([invalid_example])
            self.assertTrue(any("Failed to load image" in str(warn.message) for warn in w))
            self.assertTrue(output["image_indices"][0] is None)

    def test_empty_batch(self):
        """Test handling of empty batches"""
        with self.assertRaises(ValueError):
            self.collator([])

    def test_pad_to_multiple(self):
        """Test padding sequences to multiple of N"""
        collator = UnslothVisionDataCollator(
            processor=self.processor,
            max_seq_length=10,
            pad_to_multiple_of=8
        )
        
        batch = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}]
        output = collator(batch)
        
        # Length should be padded to next multiple of 8
        self.assertEqual(output["input_ids"].shape[1] % 8, 0)

if __name__ == '__main__':
    unittest.main() 