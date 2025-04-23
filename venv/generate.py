# from transformers import pipeline
#
# generator = pipeline("text-generation", model="gpt2-model", tokenizer="gpt2-model")
# text = generator("здравствуй", max_length=100, do_sample=True, top_k=50, temperature=0.9, top_p=0.95)
# print(text[0]["generated_text"])


from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# === 1. Загружаем обученную модель и токенизатор ===
def load_model():
    model = GPT2LMHeadModel.from_pretrained("../gpt2_from_scratch/gpt2-model")  # твоя дообученная модель
    tokenizer = GPT2TokenizerFast.from_pretrained("../gpt2_from_scratch/gpt2-model")
    model.eval()  # Перевод в режим инференса
    model.to("cpu")  # Или "cpu", если нет GPU
    return model, tokenizer

def generate_answer(input_text):
    global model, tokenizer

    # === 2. Ввод начального текста ===
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    # === 3. Генерация текста ===
    output = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        temperature=1.5,  # насколько "хаотичная" генерация
        top_k=150,         # отбираем только 50 самых вероятных токенов
        top_p=0.95,       # фильтрация по вероятностям (nucleus sampling)
        do_sample=True,   # включаем случайность
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.7,
    )

    # === 4. Декодируем и выводим результат ===
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

model, tokenizer = load_model()
