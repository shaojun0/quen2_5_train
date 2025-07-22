from torch.utils.data import Dataset
from transformers import Qwen2_5_VLProcessor
import torch


class ImageOnlyDecoderCaptioningDataset(Dataset):
    def __init__(self,dataset,processor):
        self.dataset = dataset
        self.processor :Qwen2_5_VLProcessor = processor
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

        self.output_kwargs = Qwen2_5_VLProcessorKwargs(size={"shortest_edge": 28 * 28, "longest_edge": 28 * 28 * 4})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        self.messages.append({"role":"assistant","content":data["en_cap"]})
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[text], images=[data["img"]],return_tensors="pt",**self.output_kwargs)
        labels_tokenize = self.processor.tokenizer(data["en_cap"])["input_ids"]
        label_padding_len = len(inputs["input_ids"].tolist()[0])-len(labels_tokenize)
        inputs["labels"] = torch.tensor([[IGNORE_INDEX]*label_padding_len+labels_tokenize])
        return inputs
