# -*- coding: utf-8 -*-
import json
import re
import numpy as np
import unicodedata
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import sys
import time
import math
import spacy

# Cargar modelo spaCy
nlp = spacy.load("es_core_news_sm")

# Cargar recursos
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('response_map.json', 'r') as f:
    resp2idx = json.load(f)
    distinct_responses = [None] * len(resp2idx)
    for r, i in resp2idx.items():
        distinct_responses[int(i)] = r

# Cargar modelo entrenado
final_model = tf.keras.models.load_model('best_model.keras')

# Cargar metadatos
with open('metadata.json', 'r') as f:
    metadata = json.load(f)

# Configuración
MAX_LEN = 60
OOV_TOKEN = '<OOV>'
oov_index = tokenizer.word_index[OOV_TOKEN]

# Funciones matemáticas
def evaluate_math_expression(expr):
    try:
        expr = expr.replace('×', '*').replace('x', '*').replace('÷', '/').replace('^', '**')
        expr = re.sub(r"[^0-9+\-*/.()^ ]", "", expr)
        if expr:
            result = eval(expr, {'__builtins__': None}, {'math': math})
            return float(result) if isinstance(result, (int, float)) else result
    except Exception as e:
        return f"Error al evaluar la expresión: {e}"

def contains_math_expression(text):
    patterns = [
        r'\d+\s*[\+\-\*/x×÷]\s*\d+',
        r'cuanto es (.*)\?',
        r'calcula (.*)',
        r'resultado de (.*)',
        r'\d+\s*\^\s*\d+',
        r'raiz cuadrada de \d+',
        r'\d+\s*!'
    ]
    return any(re.search(p, text.lower()) for p in patterns)

# Función para normalizar el texto
def normalize_text(text):
    doc = nlp(text.lower())
    tokens = [t.lemma_ for t in doc if t.is_alpha]
    return ' '.join(tokens)

# Función para generar respuesta
def generate_response(user_text):
    if contains_math_expression(user_text):
        expr = re.search(r'(?:cuanto es|calcula|resultado de)\s*(.*?)\??$', user_text.lower())
        math_expr = expr.group(1) if expr else user_text
        result = evaluate_math_expression(math_expr)
        if isinstance(result, str):  # Si hay un error en la evaluación
            return result
        if result is not None:
            if isinstance(result, float):
                result = int(result) if result.is_integer() else round(result, 4)
            return f"El resultado de {math_expr} es {result}"
        return "No pude calcular esa expresión matemática. ¿Podrías formularla de otra manera?"

    user_seq = tokenizer.texts_to_sequences([normalize_text(user_text)])[0]

    if not user_seq:
        return "Lo siento, no te entendí."

    oov_ratio = sum(1 for i in user_seq if i == oov_index) / len(user_seq)
    if oov_ratio > 0.4:
        return "No entendí bien. ¿Podrías decirlo de otra forma?"

    pad = pad_sequences([user_seq], maxlen=MAX_LEN, padding='post')
    pred = final_model.predict(pad, verbose=0)[0]

    max_prob = np.max(pred)
    predicted_index = np.argmax(pred)
    predicted_response = distinct_responses[predicted_index]

    # Opción: si la probabilidad es baja
    if max_prob < 0.3:
        return "Todavía estoy aprendiendo y no entendí bien. ¿Podrías explicarlo diferente?"

    # Obtener metadatos asociados a la respuesta
    response_metadata = metadata[predicted_index]
    intent = response_metadata.get("intent", "Desconocido")
    task = response_metadata.get("task", "No especificado")
    meaning = response_metadata.get("meaning", "No definido")
    examples = response_metadata.get("examples", [])

    # Construir respuesta enriquecida
    response_details = f"Intento: {intent}\nTarea: {task}\nSignificado: {meaning}"
    if examples:
        response_details += f"\nEjemplo: {examples[0]}"

    # Respuesta final
    return f"{predicted_response}\n\n{response_details}"

# Simular escritura
def simulate_typing(text, delay=0.03):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# 💬 Interacción por consola
if __name__ == "__main__":
    print("Chatbot listo. Escribe algo (o 'salir' para terminar):")
    while True:
        try:
            user_input = input("Vos: ")
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print("Pura vida, hasta luego.")
                break
            response = generate_response(user_input)
            simulate_typing("Bot: " + response)
        except KeyboardInterrupt:
            print("\nPura vida, hasta luego.")
            break
        except Exception as e:
            print(f"Error: {e}")
