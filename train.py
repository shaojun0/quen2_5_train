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
    generate_ids = model.generate(eval_dataset[0:2].input_ids, max_length=30)
    print(generate_ids)


if __name__ == '__main__':
    run_only_decoder()
