import tkinter as tk
from tkinter import scrolledtext

def request(request_text):
    # определить метод для обращения к модели через http
    return request_text


def get_request_text(text_entry, response_scrolled):
    text = str(text_entry.get("1.0", "end"))
    response_scrolled.delete("1.0", "end")
    response_scrolled.insert(tk.END, request(text))


def main():
    window = tk.Tk()
    window.title("Языковая модель")
    window.geometry("800x600")
    window.config(bg="#91D9A3")

    # окно для вывода ответа
    response_srcoled = scrolledtext.ScrolledText(window, width=60, height=11, font=("Calibri", 14))
    response_srcoled.place(x=160, y=15)

    # надпись приглашение ко вводу
    text_entry_label = tk.Label(window, text="Введите ключевую фразу для генерации текста", font=("Calibri", 14))
    text_entry_label.config(bg="#91D9A3")
    text_entry_label.place(x=215, y=280)

    # поле для ввода
    text_entry = tk.Text(window, width = 50, height=8, font=("Calibri", 14))
    text_entry.place(x=160, y=320)

    # кнопка для ввода текста
    text_entry_button = tk.Button(window, text="Получить результат", width=30, height=1, font=("Calibri", 14), borderwidth=1, relief="sunken", activebackground="#C8CFC9", command=lambda: get_request_text(text_entry, response_srcoled))
    text_entry_button.config(bg="#DAE3DB")
    text_entry_button.place(x=245, y=520)

    # логотип
    logo_image = tk.PhotoImage(file="/logo.png")
    logo_image = logo_image.subsample(8, 8)
    logo_label = tk.Label(window, image=logo_image)
    logo_label.place(x=10, y=10)


    window.mainloop()
