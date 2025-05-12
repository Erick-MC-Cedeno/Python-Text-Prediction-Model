# -*- coding: utf-8 -*-
import json
import re
import numpy as np
import unicodedata
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
import time
import sys
import pickle
from collections import OrderedDict
import spacy
import os
import math

# Cargar modelo spaCy
nlp = spacy.load("es_core_news_sm")
memory = OrderedDict()

# Configuracion
VOCAB_SIZE = 5000
EMBEDDING_DIM = 100
MAX_LEN = 60
NUM_NEURONS = 32
EPOCHS = 100
BATCH_SIZE = 16
INITIAL_LR = 2e-4
DROPOUT_RATE = 0.5
L2_RATE = 1e-4
VALIDATION_SPLIT = 0.2
KFOLDS = 5

# Funciones auxiliares
def warmup_scheduler(epoch, lr):
    if epoch < 5:
        return lr + (INITIAL_LR - 1e-5) / 5
    return lr

def normalize_text(text):
    doc = nlp(text.lower())
    tokens = [t.lemma_ for t in doc if t.is_alpha]
    return ' '.join(tokens)

def augment_texts(texts, completions):
    augmented = []
    for t in texts:
        words = t.split()
        if len(words) > 2:
            idx = np.random.randint(len(words))
            words[idx] = words[idx][::-1]
        augmented.append(' '.join(words))
    return texts + augmented, completions + completions

# Cargar datos
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)['conversations']

expanded_prompts = []
expanded_completions = []

for conv in data:
    completion = conv['completion'].strip()
    expanded_prompts.append(conv['prompt'])
    expanded_completions.append(completion)
    for pattern in conv['pattern']:
        expanded_prompts.append(pattern)
        expanded_completions.append(completion)

prompts = [normalize_text(p) for p in expanded_prompts]
prompts_aug, completions_aug = augment_texts(prompts, expanded_completions)

# Tokenizacion
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(prompts_aug)
oov_index = tokenizer.word_index[tokenizer.oov_token]
seqs = tokenizer.texts_to_sequences(prompts_aug)
padded_inputs = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')

# Preparar salidas
distinct_responses = sorted(set(completions_aug))
resp2idx = {r: i for i, r in enumerate(distinct_responses)}
y_indices = np.array([resp2idx[c] for c in completions_aug])
y_onehot = to_categorical(y_indices, num_classes=len(distinct_responses))

# Guardar tokenizer y mapeo
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('response_map.json', 'w') as f:
    json.dump(resp2idx, f)

# Modelo
def build_model():
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN, mask_zero=True),
        Bidirectional(LSTM(NUM_NEURONS, kernel_regularizer=l2(L2_RATE))),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        Dense(len(distinct_responses), activation='softmax', kernel_regularizer=l2(L2_RATE))
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

callbacks = [
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', verbose=1),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    LearningRateScheduler(warmup_scheduler)
]

# KFold
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(padded_inputs):
    X_train, X_val = padded_inputs[train_idx], padded_inputs[val_idx]
    y_train, y_val = y_onehot[train_idx], y_onehot[val_idx]
    model = build_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)

# Entrenamiento final
final_model = build_model()
final_model.fit(padded_inputs, y_onehot, validation_split=VALIDATION_SPLIT,
                epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)
final_model.save('chatbot_model_final.keras')

# -------------------- INTERACCION --------------------
# Funciones matematicas
def evaluate_math_expression(expr):
    try:
        expr = expr.replace('×', '*').replace('x', '*').replace('÷', '/').replace('^', '**')
        expr = re.sub(r"[^0-9+\-*/.()^ ]", "", expr)
        if expr:
            result = eval(expr, {'__builtins__': None}, {'math': math})
            return float(result) if isinstance(result, (int, float)) else result
    except:
        return None

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

# Cargar recursos
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('response_map.json', 'r') as f:
    resp2idx = json.load(f)
    distinct_responses = [None] * len(resp2idx)
    for r, i in resp2idx.items():
        distinct_responses[int(i)] = r

# Cargar modelo entrenado
final_model = tf.keras.models.load_model('chatbot_model_final.keras')

def generate_response(user_text):
    if contains_math_expression(user_text):
        expr = re.search(r'(?:cuanto es|calcula|resultado de)\s*(.*?)\??$', user_text.lower())
        math_expr = expr.group(1) if expr else user_text
        result = evaluate_math_expression(math_expr)
        if result is not None:
            if isinstance(result, float):
                result = int(result) if result.is_integer() else round(result, 4)
            return f"El resultado de {math_expr} es {result}"
        return "No pude calcular esa expresion matematica. Podrias formularla de otra manera?"

    user_seq = tokenizer.texts_to_sequences([normalize_text(user_text)])[0]

    if not user_seq:
        return "Lo siento, no te entendi."

    oov_ratio = sum(1 for i in user_seq if i == oov_index) / len(user_seq)
    if oov_ratio > 0.4:
        return "No entendi bien. Podrias decirlo de otra forma?"

    pad = pad_sequences([user_seq], maxlen=MAX_LEN, padding='post')
    pred = final_model.predict(pad, verbose=0)[0]

    top_indices = pred.argsort()[-3:][::-1]
    top_responses = [distinct_responses[i] for i in top_indices]

    if len(set(top_responses)) == 1:
        return "Podrias ser mas especifico para poder ayudarte mejor?"
    if np.max(pred) < 0.3:
        return "No estoy seguro de lo que quieres decir. Podrias intentar reformular?"

    return distinct_responses[np.argmax(pred)]

def simulate_typing(text, delay=0.03):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# 💬 Interacción por consola
if __name__ == "__main__":
    print("Chatbot listo. Escribí algo (o 'salir' para terminar):")
    while True:
        user_input = input("Vos: ")
        if user_input.lower() in ['salir', 'exit', 'quit']:
            print("Pura vida, hasta luego.")
            break
        response = generate_response(user_input)
        simulate_typing("Bot: " + response)
