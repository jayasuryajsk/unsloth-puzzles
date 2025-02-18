import torch
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from transformers import AutoProcessor
from typing import List, Optional, Union, Dict, Any
import warnings

class ImageResizeTransform:
    """
    Resizes PIL images or arrays to a fixed (max_image_size, max_image_size).
    """
    def __init__(self, max_image_size=224, mode="bilinear"):
        self.max_image_size = max_image_size
        interp = (T.InterpolationMode.BILINEAR 
                  if mode == "bilinear" 
                  else T.InterpolationMode.NEAREST)
        self.transform = T.Resize((max_image_size, max_image_size), interpolation=interp)

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = self.transform(image)
        return image


class UnslothVisionDataCollator:
    """
    Enhanced data collator for vision-language models that efficiently handles:
    - Text-only examples
    - Single image examples
    - Multiple images per example
    - Mixed batches with varying numbers of images
    
    Supports models like Qwen-VL, LLaMA-Vision, etc.
    """
    def __init__(
        self,
        processor: AutoProcessor,
        max_seq_length: int = 2048,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        pad_to_multiple_of: Optional[int] = 8,
        max_image_size: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        """
        Args:
            processor: HuggingFace processor (e.g. QwenProcessor, LlavaProcessor)
            max_seq_length: Maximum sequence length for text
            pad_token_id: Token ID to use for padding text
            label_pad_token_id: Token ID to use for padding labels (-100 to ignore in loss)
            pad_to_multiple_of: Optional multiple for padding sequence lengths
            max_image_size: Optional maximum image size (height/width) for resizing
            return_tensors: Return format for tensors ("pt" for PyTorch)
        """
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        
        # Set up image resizing if specified
        self.image_transform = None
        if max_image_size is not None:
            self.image_transform = ImageResizeTransform(max_image_size)
            
        # Try to get image size from model config if not specified
        elif hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
            config_size = processor.image_processor.size
            if isinstance(config_size, (tuple, list)):
                config_size = config_size["height"]
            self.image_transform = ImageResizeTransform(config_size)

    def _process_images(self, images: List[Union[Image.Image, str, bytes]]) -> List[torch.Tensor]:
        """Process a list of images, handling various input formats."""
        if not images:
            return []
            
        processed_images = []
        for img in images:
            # Convert string/bytes to PIL if needed
            if isinstance(img, (str, bytes)):
                try:
                    img = Image.open(img).convert('RGB')
                except Exception as e:
                    warnings.warn(f"Failed to load image, skipping: {e}")
                    continue
                    
            # Resize if transform is set
            if self.image_transform is not None:
                img = self.image_transform(img)
                
            # Let processor handle the rest (normalization etc)
            processed_images.append(img)
            
        if not processed_images:
            return []
            
        # Process all images in one batch for efficiency
        try:
            processed = self.processor(images=processed_images, return_tensors=self.return_tensors)
            return [processed.pixel_values[i] for i in range(len(processed_images))]
        except Exception as e:
            warnings.warn(f"Image processing failed: {e}")
            return []

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples, handling text and images.
        
        Args:
            examples: List of dicts with keys like "input_ids", "attention_mask", 
                     "labels", "images" or "pixel_values"
                     
        Returns:
            Batch dict with collated tensors
        """
        # Handle text inputs
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        # Track images for each example
        all_pixel_values = []
        image_indices = []  # Maps each example to its images' indices
        current_image_idx = 0
        
        for ex in examples:
            # Truncate if needed
            if len(ex["input_ids"]) > self.max_seq_length:
                ex["input_ids"] = ex["input_ids"][:self.max_seq_length]
                ex["attention_mask"] = ex["attention_mask"][:self.max_seq_length]
                if "labels" in ex:
                    ex["labels"] = ex["labels"][:self.max_seq_length]
                    
            # Convert to tensors
            input_ids_list.append(torch.tensor(ex["input_ids"], dtype=torch.long))
            attention_mask_list.append(torch.tensor(ex["attention_mask"], dtype=torch.long))
            if "labels" in ex:
                labels_list.append(torch.tensor(ex["labels"], dtype=torch.long))
            
            # Handle images - support multiple sources and formats
            example_images = []
            
            # Direct image data
            if "images" in ex:
                if isinstance(ex["images"], (list, tuple)):
                    example_images.extend(ex["images"])
                else:
                    example_images.append(ex["images"])
                    
            # Pre-processed pixel values
            elif "pixel_values" in ex:
                if isinstance(ex["pixel_values"], (list, tuple)):
                    example_images.extend(ex["pixel_values"])
                else:
                    example_images.append(ex["pixel_values"])
                    
            # Process images if any found
            if example_images:
                processed_images = self._process_images(example_images)
                if processed_images:  # Only add if processing succeeded
                    all_pixel_values.extend(processed_images)
                    image_indices.append((current_image_idx, 
                                        current_image_idx + len(processed_images)))
                    current_image_idx += len(processed_images)
                else:
                    image_indices.append(None)  # Mark as no images
            else:
                image_indices.append(None)  # No images for this example
        
        # Pad sequences
        input_ids_padded = pad_sequence(
            input_ids_list, 
            batch_first=True,
            padding_value=self.pad_token_id
        )
        attention_mask_padded = pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0
        )
        
        # Handle labels if present
        if labels_list:
            labels_padded = pad_sequence(
                labels_list,
                batch_first=True,
                padding_value=self.label_pad_token_id
            )
        else:
            labels_padded = None
            
        # Optional length padding
        if self.pad_to_multiple_of and self.pad_to_multiple_of > 1:
            pad_len = (self.pad_to_multiple_of - 
                      input_ids_padded.size(1) % self.pad_to_multiple_of)
            if pad_len != self.pad_to_multiple_of:
                input_ids_padded = torch.nn.functional.pad(
                    input_ids_padded,
                    (0, pad_len),
                    value=self.pad_token_id
                )
                attention_mask_padded = torch.nn.functional.pad(
                    attention_mask_padded,
                    (0, pad_len),
                    value=0
                )
                if labels_padded is not None:
                    labels_padded = torch.nn.functional.pad(
                        labels_padded,
                        (0, pad_len),
                        value=self.label_pad_token_id
                    )
        
        # Build output batch
        batch = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
        }
        
        if labels_padded is not None:
            batch["labels"] = labels_padded
            
        if all_pixel_values:
            batch["pixel_values"] = torch.stack(all_pixel_values)
            batch["image_indices"] = image_indices
            
        return batch
