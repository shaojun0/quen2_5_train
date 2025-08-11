from torch.utils.data import Dataset
from transformers import Qwen2_5_VLProcessor
import torch
from transformers.models.qwen2_5_vl.modular_qwen2_5_vl import Qwen2_5_VLProcessorKwargs
import numpy as np

IGNORE_INDEX=-100

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
        while True:
            data = self.dataset[idx]
            try:
                self.messages.append({"role":"assistant","content":data["en_cap"]})
                text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=False)
                inputs = self.processor(text=[text], images=[data["img"]],return_tensors="pt",**self.output_kwargs)
                labels_tokenize = self.processor.tokenizer(data["en_cap"])["input_ids"]
                label_padding_len = len(inputs["input_ids"].tolist()[0])-len(labels_tokenize)
                inputs["labels"] = torch.tensor([[IGNORE_INDEX]*label_padding_len+labels_tokenize])
                break
            except Exception as e:
                idx = np.random.randint(self.__len__())
                print("发生错误跳过文件")
        return inputs


class Florence2Dataset(Dataset):
    def __init__(self,dataset,processor,return_org_image:bool=False):
        self.dataset = dataset
        self.processor = processor
        self.current = 0
        self.return_org_image = return_org_image

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        inputs = self.processor(text="<OD>", images=[data["img"]], return_tensors="pt",do_convert_rgb=True)
        inputs["labels"] = self.processor.tokenizer([data["en_cap"]], return_tensors="pt")["input_ids"]
        if self.return_org_image:
            inputs["org_images"] = data["img"]
        return inputs

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.__len__():
            raise StopIteration
        self.current += 1
        return self.__getitem__(self.current-1)