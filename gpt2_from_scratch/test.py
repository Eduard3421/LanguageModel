# from datasets import load_dataset
#
# dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
# with open("corpus4.txt", "w", encoding="utf-8") as f:
#     for item in dataset:
#         f.write(item["text"] + "\n")

# def clean_text_file(input_path, output_path):
#     with open(input_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#
#     cleaned_lines = []
#     for line in lines:
#         # Удаление неразрывных пробелов и html-сущностей
#         cleaned = line.replace('\xa0', ' ').replace('&nbsp;', ' ')
#         # Удаление лишних пробелов в конце и начале
#         cleaned = cleaned.strip()
#         # Добавляем только непустые строки
#         if cleaned:
#             cleaned_lines.append(cleaned)
#
#     # Запись очищенного текста
#     with open(output_path, 'w', encoding='utf-8') as f:
#         for line in cleaned_lines:
#             f.write(line + '\n')
#
#
# # Пример использования
# clean_text_file("corpus4.txt", "corpus5.txt")

# def clean_text_file(input_path, output_path):
#     with open(input_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#
#     cleaned_lines = []
#     for line in lines:
#         cleaned = line \
#             .replace('\xa0', ' ') \
#             .replace('&nbsp;', ' ') \
#             .replace('\u00AD', '') \
#             .replace('&shy;', '')
#
#         cleaned = cleaned.strip()
#         if cleaned:
#             cleaned_lines.append(cleaned)
#
#     with open(output_path, 'w', encoding='utf-8') as f:
#         for line in cleaned_lines:
#             f.write(line + '\n')
#
#
# # Пример использования
# clean_text_file("corpus5.txt", "corpus4.txt")

# import re
#
# input_file = "corpus4.txt"
# output_file = "corpus4_fixed.txt"
# max_line_length = 200
#
# with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
#     for line in fin:
#         line = line.strip()
#         if len(line) <= max_line_length:
#             fout.write(line + "\n")
#         else:
#             # Попробуем сначала разбить по предложениям
#             sentences = re.split(r'(?<=[.!?]) +', line)
#             for sentence in sentences:
#                 if len(sentence) <= max_line_length:
#                     fout.write(sentence.strip() + "\n")
#                 else:
#                     # Если даже предложение длинное — режем по кускам
#                     for i in range(0, len(sentence), max_line_length):
#                         fout.write(sentence[i:i+max_line_length].strip() + "\n")

# def truncate_file(input_file_path, output_file_path, fraction= 1 / 20):
#     # Открываем файл для чтения
#     with open(input_file_path, 'r', encoding='utf-8') as f:
#         # Получаем общее количество строк в файле
#         lines = f.readlines()
#
#     # Определяем сколько строк оставить (3/10 от общего числа строк)
#     num_lines_to_keep = int(len(lines) * fraction)
#
#     # Записываем нужное количество строк в новый файл
#     with open(output_file_path, 'w', encoding='utf-8') as f:
#         f.writelines(lines[:num_lines_to_keep])
#
#
# # Применение
# input_file = "corpus5.txt"
# output_file = "corpus4_fixed.txt"
# truncate_file(input_file, output_file)

import os

os.environ["HF_DATASETS_CACHE"] = "D:\\huggingface_cache"

from datasets import load_dataset

literature = load_dataset("cointegrated/taiga_stripped_proza", split="train")

print(literature[0])
