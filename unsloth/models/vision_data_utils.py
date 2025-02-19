import torch
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from transformers import AutoProcessor
from typing import List, Optional, Union, Dict, Any, Tuple
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
    - Text-only examples (no images)
    - Single image examples
    - Multiple images per example
    - Mixed batches with varying numbers of images
    - Automatic image resizing and preprocessing
    
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
        skip_image_processor_errors: bool = True,
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
            skip_image_processor_errors: If True, skip images that fail to process instead of raising error
        """
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.skip_image_processor_errors = skip_image_processor_errors
        
        # Set up image resizing if specified
        self.image_transform = None
        if max_image_size is not None:
            self.image_transform = ImageResizeTransform(max_image_size)
        # Try to get image size from model config if not specified
        elif hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
            config_size = processor.image_processor.size
            if isinstance(config_size, (tuple, list)):
                config_size = config_size["height"] if isinstance(config_size, dict) else config_size[0]
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
                    if not self.skip_image_processor_errors:
                        raise
                    warnings.warn(f"Failed to load image, skipping: {e}")
                    continue
                    
            # Ensure image is in RGB mode
            if not isinstance(img, Image.Image):
                try:
                    img = Image.fromarray(img).convert('RGB')
                except Exception as e:
                    if not self.skip_image_processor_errors:
                        raise
                    warnings.warn(f"Failed to convert image to PIL, skipping: {e}")
                    continue
            elif img.mode != 'RGB':
                img = img.convert('RGB')
                    
            # Resize if transform is set
            if self.image_transform is not None:
                try:
                    img = self.image_transform(img)
                except Exception as e:
                    if not self.skip_image_processor_errors:
                        raise
                    warnings.warn(f"Failed to resize image, skipping: {e}")
                    continue
                
            processed_images.append(img)
            
        if not processed_images:
            return []
            
        # Process all images in one batch for efficiency
        try:
            processed = self.processor(images=processed_images, return_tensors=self.return_tensors)
            return [processed.pixel_values[i] for i in range(len(processed_images))]
        except Exception as e:
            if not self.skip_image_processor_errors:
                raise
            warnings.warn(f"Image processing failed: {e}")
            return []

    def _prepare_text_inputs(self, example: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Prepare text inputs for a single example."""
        # Handle cases where input_ids might be missing (e.g., text needs to be tokenized)
        if "input_ids" not in example:
            if "text" in example:
                # Tokenize text if not already tokenized
                inputs = self.processor.tokenizer(
                    example["text"],
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors=None  # Return list
                )
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
            else:
                raise ValueError("Example must contain either 'input_ids' or 'text'")
        else:
            input_ids = example["input_ids"]
            attention_mask = example.get("attention_mask", [1] * len(input_ids))

        # Truncate if needed
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Handle labels if present
        labels = None
        if "labels" in example:
            labels = example["labels"]
            if len(labels) > self.max_seq_length:
                labels = labels[:self.max_seq_length]
            labels = torch.tensor(labels, dtype=torch.long)
            
        return input_ids, attention_mask, labels

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples, handling text and images.
        
        Args:
            examples: List of dicts with keys:
                - Required: either "input_ids" or "text"
                - Optional: "attention_mask", "labels", "images"
                     
        Returns:
            Batch dict with collated tensors
        """
        if not examples:
            raise ValueError("Cannot collate empty batch")
            
        # Process text inputs
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        # Track images
        all_pixel_values = []
        image_indices = []  # List of (start_idx, end_idx) or None for text-only examples
        current_image_idx = 0
        
        for ex in examples:
            # Handle text inputs
            input_ids, attention_mask, labels = self._prepare_text_inputs(ex)
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            if labels is not None:
                labels_list.append(labels)
            
            # Handle images - support both "images" and "pixel_values" keys
            if "pixel_values" in ex:
                # Direct pixel values (already processed)
                if torch.is_tensor(ex["pixel_values"]):
                    pixels = [ex["pixel_values"]]
                else:
                    pixels = ex["pixel_values"]
                all_pixel_values.extend(pixels)
                image_indices.append((current_image_idx, 
                                   current_image_idx + len(pixels)))
                current_image_idx += len(pixels)
            elif "images" in ex:
                # Raw images needing processing
                if isinstance(ex["images"], (list, tuple)):
                    processed = self._process_images(ex["images"])
                else:
                    processed = self._process_images([ex["images"]])
                    
                if processed:
                    all_pixel_values.extend(processed)
                    image_indices.append((current_image_idx, 
                                       current_image_idx + len(processed)))
                    current_image_idx += len(processed)
                else:
                    # No images processed successfully
                    image_indices.append(None)
            else:
                # Text-only example
                image_indices.append(None)
        
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
        
        if labels_list:
            labels_padded = pad_sequence(
                labels_list,
                batch_first=True,
                padding_value=self.label_pad_token_id
            )
        else:
            labels_padded = None
            
        # Optional length padding to multiple
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
