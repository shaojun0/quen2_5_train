from datasets import load_dataset
from transformers import TrainingArguments, Qwen2_5_VLForConditionalGeneration, AutoProcessor, Trainer, \
    Qwen2_5_VLProcessor, AutoModelForCausalLM

from collector import qwen_2_5_collator, florence2_collator
from dataset import ImageOnlyDecoderCaptioningDataset, Florence2Dataset
import json


def run_only_decoder(is_scnet=False):
    dataset_files = "Obscure-Entropy/ImageCaptioning_SmallParquets_old"
    model_path = "models/Qwen2.5-VL-3B-Instruct"
    if is_scnet:
        dataset_files = "/public/home/scnvewz0f6/SothisAI/dataset/ExternalSource/ImageCaptioning_SmallParquets" \
                        "/main/ImageCaptioning_SmallParquets"
        model_path = "/work/home/scnbfowvjz/SothisAI/model/Aihub/Qwen2.5-VL-3B-Instruct/main/Qwen2.5-VL-3B-Instruct"
    train_dataset = load_dataset(dataset_files,split="train[:1%]")
    eval_dataset = load_dataset(dataset_files,split="train[1%:2%]")
    output_dir = "outputs"
    training_args = TrainingArguments(output_dir=output_dir,
                                      per_device_train_batch_size=4,
                                      per_device_eval_batch_size=16,
                                      num_train_epochs=3,
                                      save_safetensors=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    train_dataset = ImageOnlyDecoderCaptioningDataset(train_dataset, processor)
    val_dataset = ImageOnlyDecoderCaptioningDataset(eval_dataset, processor)
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      data_collator=qwen_2_5_collator(processor.tokenizer))
    trainer.train()
    trainer.save_model(output_dir)


def run_only_decoder_deepspeed(is_scnet=True):
    dataset_files = "Obscure-Entropy/ImageCaptioning_SmallParquets_old"
    model_path = "models/Qwen2.5-VL-3B-Instruct"
    if is_scnet:
        dataset_files = "/work/home/scnbfowvjz/SothisAI/dataset/Aihub/ImageCaptioning_SmallParquets/main/ImageCaptioning_SmallParquets_old/data"
        model_path = "/work/home/scnbfowvjz/SothisAI/model/Aihub/Qwen2.5-VL-3B-Instruct/main/Qwen2.5-VL-3B-Instruct"
    train_dataset = load_dataset(dataset_files, split="train[:1%]")
    eval_dataset = load_dataset(dataset_files, split="train[1%:2%]")
    output_dir = "outputs"
    deep_speed_path = "DeepSpeedExamples/training/autotuning/hf/dsconfigs/ds_config_z2.json"
    with open(deep_speed_path, encoding="utf-8") as f:
        deep_speed_config = json.load(f)
    training_args = TrainingArguments(output_dir=output_dir,
                                      per_device_train_batch_size=1,
                                      per_device_eval_batch_size=8,
                                      num_train_epochs=3,
                                      save_safetensors=True,
                                      deepspeed=deep_speed_config,
                                      gradient_accumulation_steps=4,
                                      fp16=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

    train_dataset = ImageOnlyDecoderCaptioningDataset(train_dataset, processor)
    val_dataset = ImageOnlyDecoderCaptioningDataset(eval_dataset, processor)
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      data_collator=qwen_2_5_collator(processor.tokenizer))
    trainer.train()
    trainer.save_model(output_dir)
    trainer.evaluate()

def run_florence2():
    train_dataset = load_dataset("data/translate/total",split="train[:95%]",cache_dir="./.cache")
    eval_dataset = load_dataset("data/translate/total",split="train[95%:]",cache_dir="./.cache")
    model_path = "models/Florence-2-large-Chinese"
    output_dir = "outputs"
    deep_speed_path = "DeepSpeedExamples/training/autotuning/hf/dsconfigs/ds_config_z2.json"
    training_args = TrainingArguments(output_dir=output_dir,
                                      per_device_train_batch_size=4,
                                      per_device_eval_batch_size=8,
                                      num_train_epochs=3,
                                      save_safetensors=True,
                                      deepspeed=deep_speed_path,
                                      fp16=True,
                                      gradient_accumulation_steps=16,
                                      learning_rate=1e-6)
    model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
    train_dataset = Florence2Dataset(train_dataset, processor)
    val_dataset = Florence2Dataset(eval_dataset, processor)
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      data_collator=florence2_collator(processor.tokenizer))
    trainer.train()
    trainer.save_model(output_dir)
    trainer.evaluate()


if __name__ == '__main__':
    run_florence2()
