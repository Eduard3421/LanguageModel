from tokenizers.implementations import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files="corpus4_fixed.txt",
    vocab_size=20000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

tokenizer.save_model("tokenizer")
