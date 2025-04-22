from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer")
tokenizer.pad_token = "<pad>"

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_ctx=512,
    n_embd=256,
    n_layer=6,
    n_head=8,
    pad_token_id=tokenizer.pad_token_id
)

model = GPT2LMHeadModel(config)

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="corpus.txt",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="./gpt2-model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()

trainer.save_model("gpt2-model")
tokenizer.save_pretrained("gpt2-model")
