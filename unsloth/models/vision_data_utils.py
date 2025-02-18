import torch
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from transformers import AutoProcessor

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


class MixedVisionTextCollator:
    """
    Merges text-only and text+image examples in the same batch.

    Usage:
      collator = MixedVisionTextCollator(processor, ...)
      batch = collator(list_of_samples)
    """
    def __init__(
        self, 
        processor=None,
        max_seq_length=1024,
        pad_token_id=0,
        pad_to_multiple_of=8,
    ):
        """
        :param processor: e.g. Qwen-VL, LlamaVision, etc.
        :param max_seq_length: max # tokens
        :param pad_token_id: token ID for text padding
        :param pad_to_multiple_of: optional multiple for length padding 
        """
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        input_ids_list = []
        attention_mask_list = []
        pixel_values_list = []
        has_image_flags = []

        for ex in batch:
            # text fields
            if len(ex["input_ids"]) > self.max_seq_length:
                ex["input_ids"] = ex["input_ids"][: self.max_seq_length]
                ex["attention_mask"] = ex["attention_mask"][: self.max_seq_length]

            input_ids_list.append(torch.tensor(ex["input_ids"], dtype=torch.long))
            attention_mask_list.append(torch.tensor(ex["attention_mask"], dtype=torch.long))

            # image field
            if "pixel_values" in ex:
                # if user already has a Tensor or PIL
                if isinstance(ex["pixel_values"], torch.Tensor):
                    pixel_values_list.append(ex["pixel_values"])
                else:
                    # Use self.processor if we want e.g. normalization
                    # Or just store the raw
                    if self.processor is not None:
                        pixel = self.processor(images=ex["pixel_values"], return_tensors="pt").pixel_values[0]
                    else:
                        pixel = ex["pixel_values"]  # e.g. a PIL or array
                    pixel_values_list.append(pixel)
                has_image_flags.append(True)
            else:
                pixel_values_list.append(None)
                has_image_flags.append(False)

        # pad text
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        # optionally pad dimension
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 1:
            total_len = input_ids_padded.shape[1]
            remainder = total_len % self.pad_to_multiple_of
            if remainder != 0:
                pad_size = self.pad_to_multiple_of - remainder
                pad_ids = torch.full((input_ids_padded.size(0), pad_size), self.pad_token_id, dtype=torch.long)
                input_ids_padded = torch.cat([input_ids_padded, pad_ids], dim=1)
                pad_mask = torch.zeros((attention_mask_padded.size(0), pad_size), dtype=torch.long)
                attention_mask_padded = torch.cat([attention_mask_padded, pad_mask], dim=1)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "pixel_values_list": pixel_values_list,
            "has_image_flags": has_image_flags,
        }
