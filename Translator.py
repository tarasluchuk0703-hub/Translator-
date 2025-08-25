from flask import Flask, render_template, request
from transformers import pipeline, MarianMTModel, MarianTokenizer
import re
import socket
import logging

app = Flask(__name__)

# === MarianMT models ===
model_en_to_uk = 'Helsinki-NLP/opus-mt-en-uk'
marian_en_to_uk = MarianMTModel.from_pretrained(model_en_to_uk)
tokenizer_en_to_uk = MarianTokenizer.from_pretrained(model_en_to_uk)

model_uk_to_en = 'Helsinki-NLP/opus-mt-uk-en'
translator_ukrainian_to_english = pipeline(task="translation", model=model_uk_to_en)

# === Idioms dictionaries ===
idioms_en_to_uk = {
    "break a leg": "ні пуху, ні пера",
    "fall for it": "повестися",
    "fell for it": "повівся",
    "falling for it": "ведеться",
    "falls for it": "ведеться",
    "kick the bucket": "відкинути копита",
    "spill the beans": "розкрити всі карти",
    "under the weather": "погано себе почувати",
    "hit the sack": "лягти спати",
    "costs an arm and a leg": "коштує ціле багатство",
    "call it a day": "закінчити роботу",
    "cut corners": "економити на якості",
    "miss the boat": "втратити шанс",
    "piece of cake": "дуже легко",
    "pull someone's leg": "жартувати",
    "go Dutch": "заплатити кожен за себе",
    "once in a blue moon": "раз на сто років"
}
idioms_uk_to_en = {v: k for k, v in idioms_en_to_uk.items()}

# === Точні приклади ідіоматичних речень ===
known_sentences = {
    "before the big performance, the director told the actors to break a leg.":
        "перед великим виступом режисер побажав акторам ні пуху, ні пера.",
    "when we went out for dinner, we decided to go dutch so that no one would have to cover the whole bill.":
        "коли ми пішли вечеряти, ми вирішили платити кожен за себе, щоб нікому не довелося оплачувати весь рахунок."
}

# === Визначення мови тексту ===
def detect_language(text):
    ukrainian_letters = set("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя")
    russian_letters = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    english_letters = set("abcdefghijklmnopqrstuvwxyz")

    cleaned = re.sub(r'[^a-zA-Zа-яА-ЯёЁіІїЇєЄґҐ]', '', text.lower())
    counts = {
        "uk": sum(c in ukrainian_letters for c in cleaned),
        "ru": sum(c in russian_letters for c in cleaned),
        "en": sum(c in english_letters for c in cleaned)
    }
    return max(counts, key=counts.get)

# === Заміна ідіом ===
def replace_idioms_with_ukr(text):
    for idiom, ukr_eq in idioms_en_to_uk.items():
        pattern = re.compile(rf"\b{re.escape(idiom)}\b", re.IGNORECASE)
        text = pattern.sub(ukr_eq, text)
    return text

def replace_idioms_with_en(text):
    for ukr_eq, idiom in idioms_uk_to_en.items():
        pattern = re.compile(rf"\b{re.escape(ukr_eq)}\b", re.IGNORECASE)
        text = pattern.sub(idiom, text)
    return text

# === Переклад ===
def translate_en_to_uk(text):
    text_clean = text.strip().lower()
    if text_clean in known_sentences:
        return known_sentences[text_clean]

    replaced = replace_idioms_with_ukr(text)
    inputs = tokenizer_en_to_uk(replaced, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = marian_en_to_uk.generate(**inputs)
    translation = tokenizer_en_to_uk.decode(translated_tokens[0], skip_special_tokens=True)

    if detect_language(translation) == "ru":
        return "⚠ Модель переклала не українською. Спробуйте змінити формулювання або подати коротше речення."

    return translation

def translate_uk_to_en(text):
    if not text.strip():
        return ""
    replaced = replace_idioms_with_en(text)
    result = translator_ukrainian_to_english(replaced, max_length=1500)
    return result[0]['translation_text']

# === Flask routes ===
@app.route('/')
def index():
    return render_template('Darkib.html')

@app.route('/translate', methods=['POST'])
def translate():
    input_text = request.form['input_text']
    direction = request.form['direction']

    if direction == 'en_to_uk':
        output_text = translate_en_to_uk(input_text)
    else:
        output_text = translate_uk_to_en(input_text)

    return render_template('Darkib.html', input_text=input_text, output_text=output_text, direction=direction)

# === Server run logic ===
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    def find_free_port(start_port=5001, max_tries=10):
        port = start_port
        for _ in range(max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', port)) != 0:
                    return port
            port += 1
        raise RuntimeError("No free port found in range.")

    try:
        chosen_port = find_free_port()
        logging.info(f"Flask app running on port {chosen_port}")
        app.run(host='0.0.0.0', port=chosen_port, debug=False)
    except Exception as e:
        logging.critical(f"Flask failed to start: {e}")
