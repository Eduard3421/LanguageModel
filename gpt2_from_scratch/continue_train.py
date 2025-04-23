import os

# Указываем папку для кэша Hugging Face datasets
os.environ["HF_DATASETS_CACHE"] = "D:/huggingface_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache/models"
os.environ["TMPDIR"] = "D:/huggingface_cache/tmp"


from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset


# Загрузка токенизатора и модели
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-model")
tokenizer.pad_token = "<pad>"
model = GPT2LMHeadModel.from_pretrained("gpt2-model")

# Загрузка датасета
literature = load_dataset("cointegrated/taiga_stripped_proza", split="train", cache_dir="D:/huggingface_cache/datasets")

# Обработка текста перед токенизацией
def tokenize_function(examples):
    # Очистка всех текстов в батче
    texts = [text.replace("\n", " ").replace("--", " ") for text in examples["text"]]
    return tokenizer(texts, truncation=True, max_length=512, padding="max_length")

# Токенизация
# tokenized_dataset = literature.map(tokenize_function, batched=True, remove_columns=["sample"])

tokenized_dataset = literature.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "file"]
)

# Остальной код без изменений
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./gpt2-model",
    overwrite_output_dir=False,
    num_train_epochs=8,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    save_safetensors=False,
    fp16=True,
    max_steps=2000000,
    save_strategy="steps",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train(resume_from_checkpoint="./gpt2-model/checkpoint-1205000")
trainer.save_model("gpt2-model")
tokenizer.save_pretrained("gpt2-model")
