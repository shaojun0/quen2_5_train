from datasets import load_dataset
from transformers import TrainingArguments, Qwen2_5_VLForConditionalGeneration, AutoProcessor, Trainer

from collector import qwen_2_5_collator
from dataset import ImageOnlyDecoderCaptioningDataset


def run_only_decoder():
    train_dataset = load_dataset("Obscure-Entropy/ImageCaptioning_EN-HU",split="train[:90%]")
    eval_dataset = load_dataset("Obscure-Entropy/ImageCaptioning_EN-HU",split="train[90%:]")
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    output_dir = "outputs"
    training_args = TrainingArguments(output_dir=output_dir,
                                      per_device_train_batch_size=32,
                                      per_device_eval_batch_size=64,
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


def run_only_decoder_deepspeed():
    torch.npu.config.allow_internal_format=False
    torch.npu.set_compile_mode(jit_compile=False)
    train_dataset = load_dataset("data/org_large/acc",split="train[:1%]")
    eval_dataset = load_dataset("data/org_large/acc",split="train[1%:2%]")
    model_path = "models/Qwen2.5-VL-3B-Instruct"
    output_dir = "outputs"
    deep_speed_path = "DeepSpeedExamples/training/autotuning/hf/dsconfigs/ds_config_fp16_z3.json"
    training_args = TrainingArguments(output_dir=output_dir,
                                      per_device_train_batch_size=2,
                                      per_device_eval_batch_size=8,
                                      num_train_epochs=3,
                                      save_safetensors=True,
                                      deepspeed=deep_speed_path,
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


if __name__ == '__main__':
    run_only_decoder_deepspeed()
