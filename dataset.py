from torch.utils.data import Dataset
from transformers import Qwen2_5_VLProcessor
import torch


class ImageOnlyDecoderCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor: Qwen2_5_VLProcessor = processor
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[data["img"]], return_tensors="pt")
        inputs["position_ids"] = (
            torch.arange(0, inputs["input_ids"].size(1)).view(1, -1).unsqueeze(0).expand(3, -1, -1))
        inputs["labels"] = self.processor(text=[data["en_cap"]], return_tensors="pt")["input_ids"]
        return inputs
