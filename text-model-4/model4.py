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
from sklearn.neighbors import NearestNeighbors
import time
import sys
import pickle
from collections import OrderedDict
import spacy
import os
import re
import math

# Cargar modelo spaCy
nlp = spacy.load("es_core_news_sm")
memory = OrderedDict()

# Configuración
VOCAB_SIZE = 5000
EMBEDDING_DIM = 100
MAX_LEN = 60
NUM_NEURONS = 32
EPOCHS = 100
BATCH_SIZE = 16
INITIAL_LR = 2e-4
DROPOUT_RATE = 0.5
L2_RATE = 1e-4
JACCARD_THRESH = 0.4
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

def jaccard_similarity(a, b):
    inter = a & b
    uni = a | b
    return len(inter) / len(uni) if uni else 0

def extract_keywords(text):
    return set(normalize_text(text).split())

# Cargar datos con estructura corregida
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)['conversations']

# Generar dataset expandido
expanded_prompts = []
expanded_completions = []
tasks = []
meanings = []

for conv in data:
    main_prompt = conv['prompt']
    completion = conv['completion'].strip()
    patterns = conv['pattern']
    task = conv.get('task', '')  # Usar get() para manejar posibles campos faltantes
    meaning = conv.get('meaning', '')
    
    # Agregar prompt principal
    expanded_prompts.append(main_prompt)
    expanded_completions.append(completion)
    tasks.append(task)
    meanings.append(meaning)
    
    # Agregar todos los patterns como variantes
    for pattern in patterns:
        expanded_prompts.append(pattern)
        expanded_completions.append(completion)
        tasks.append(task)
        meanings.append(meaning)

# Procesamiento de texto
prompts = [normalize_text(p) for p in expanded_prompts]
prompt_sets = [set(p.split()) for p in prompts]
prompts_aug, completions_aug = augment_texts(prompts, expanded_completions)

# Tokenización
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

# Guardar recursos
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('response_map.json', 'w') as f:
    json.dump(resp2idx, f)

# Construcción del modelo
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

# Callbacks
callbacks = [
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', verbose=1),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    LearningRateScheduler(warmup_scheduler)
]

# Entrenamiento con K-Fold
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(padded_inputs):
    X_train, X_val = padded_inputs[train_idx], padded_inputs[val_idx]
    y_train, y_val = y_onehot[train_idx], y_onehot[val_idx]
    model = build_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=callbacks, verbose=2)

# Entrenamiento final
final_model = build_model()
final_model.fit(padded_inputs, y_onehot, validation_split=VALIDATION_SPLIT,
                epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)
final_model.save('chatbot_model_final.keras')

# Preparar embeddings para búsqueda semántica
embedding_layer = final_model.layers[0]
embedding_matrix = embedding_layer.get_weights()[0]

def get_average_embedding(seq):
    vectors = [embedding_matrix[idx] for idx in seq if idx != 0 and idx < VOCAB_SIZE]
    return np.mean(vectors, axis=0) if vectors else np.zeros(EMBEDDING_DIM)

prompt_embeddings = np.array([get_average_embedding(seq)
                              for seq in tokenizer.texts_to_sequences(prompts_aug)])

nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
nn_model.fit(prompt_embeddings)

with open('nn_model.pkl', 'wb') as f:
    pickle.dump(nn_model, f)
np.save('prompt_embeddings.npy', prompt_embeddings)

# Sistema de respuesta mejorado
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('response_map.json', 'r') as f:
    resp2idx = json.load(f)
    distinct_responses = [None] * len(resp2idx)
    for r, i in resp2idx.items():
        distinct_responses[int(i)] = r

def reason_before_response(user_text):
    start_time = time.time()
    best_score = 0
    best_response = None
    kw = extract_keywords(user_text)

    # Buscar en memoria primero
    for past_input, past_response in reversed(memory.items()):
        if time.time() - start_time > 5:
            break
        past_kw = extract_keywords(past_input)
        score = jaccard_similarity(kw, past_kw)
        if score > best_score and len(kw & past_kw) >= 2:
            best_score = score
            best_response = past_response

    # Buscar en los prompts y patterns
    for i, prompt_kw in enumerate(prompt_sets):
        if time.time() - start_time > 5:
            break
        score = jaccard_similarity(kw, prompt_kw)
        if score > best_score and len(kw & prompt_kw) >= 2:
            best_score = score
            best_response = expanded_completions[i]

    if best_score >= JACCARD_THRESH:
        return best_response
    return None

def generate_response(user_text):
    # Primero intentar con razonamiento basado en similitud
    response = reason_before_response(user_text)
    if response:
        return response

    # Cargar modelo de nearest neighbors
    with open('nn_model.pkl', 'rb') as f:
        nn_model = pickle.load(f)
    prompt_embeddings = np.load('prompt_embeddings.npy')

    user_seq = tokenizer.texts_to_sequences([normalize_text(user_text)])[0]
    user_embed = get_average_embedding(user_seq).reshape(1, -1)
    distances, indices = nn_model.kneighbors(user_embed)

    # Si hay alta similitud semántica, usar esa respuesta
    if distances[0][0] < 0.3:
        return completions_aug[indices[0][0]]

    # Manejar texto vacío
    if not user_seq:
        return "Lo siento, no te entendí."

    # Manejar mucho vocabulario desconocido
    oov_ratio = sum(1 for i in user_seq if i == oov_index) / len(user_seq)
    if oov_ratio > 0.4:
        return "No entendí bien. ¿Podrías decirlo de otra forma?"

    # Predecir con el modelo
    pad = pad_sequences([user_seq], maxlen=MAX_LEN, padding='post')
    pred = final_model.predict(pad, verbose=0)[0]

    # Analizar las mejores predicciones
    top_indices = pred.argsort()[-3:][::-1]
    top_responses = [distinct_responses[i] for i in top_indices]
    
    # Si hay ambigüedad en las predicciones
    if len(set(top_responses)) == 1:
        return "¿Podrías ser más específico para poder ayudarte mejor?"

    # Si la confianza es muy baja
    if np.max(pred) < 0.3:
        return "No estoy seguro de lo que quieres decir. ¿Podrías intentar reformular?"

    # Devolver la respuesta con mayor probabilidad
    idx = np.argmax(pred)
    return distinct_responses[idx]

def simulate_typing(text, delay=0.03):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(delay)
    print()


    # Agregar estas funciones al inicio del código
def evaluate_math_expression(expr):
    """Evalúa una expresión matemática segura"""
    try:
        # Reemplazar palabras comunes de operaciones
        expr = expr.replace('×', '*').replace('x', '*').replace('÷', '/')
        expr = expr.replace('^', '**').replace('por', '*').replace('entre', '/')
        
        # Eliminar caracteres no permitidos (seguridad)
        safe_expr = re.sub(r"[^0-9+\-*/.()^ ]", "", expr)
        
        # Evaluar solo si la expresión limpia no está vacía
        if safe_expr:
            result = eval(safe_expr, {'__builtins__': None}, {'math': math})
            return float(result) if isinstance(result, (int, float)) else result
    except:
        return None

def contains_math_expression(text):
    """Detecta si el texto contiene una expresión matemática"""
    math_patterns = [
        r'\d+\s*[\+\-\*\/x×÷]\s*\d+',  # Operaciones básicas
        r'cuánto es (.*)\?',             # Preguntas directas
        r'calcula (.*)',                 # Solicitudes de cálculo
        r'resultado de (.*)',            # Peticiones de resultado
        r'\d+\s*\^\s*\d+',               # Exponentes
        r'raíz cuadrada de \d+',         # Raíces
        r'\d+\s*\!'                      # Factoriales
    ]
    return any(re.search(pattern, text.lower()) for pattern in math_patterns)

# Modificar la función generate_response para incluir el manejo matemático
def generate_response(user_text):
    # Primero verificar si es una pregunta matemática
    if contains_math_expression(user_text):
        # Extraer la expresión matemática
        expr = re.search(r'(?:cuánto es|calcula|resultado de)\s*(.*?)\??$', 
                         user_text.lower())
        if expr:
            math_expr = expr.group(1)
        else:
            math_expr = user_text
        
        result = evaluate_math_expression(math_expr)
        
        if result is not None:
            # Formatear el resultado
            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 4)
            
            return f"El resultado de {math_expr} es {result}"
        else:
            return "No pude calcular esa expresión matemática. ¿Podrías formularla de otra manera?"
    
    # Luego intentar con razonamiento basado en similitud
    response = reason_before_response(user_text)
    if response:
        return response

    # Cargar modelo de nearest neighbors
    with open('nn_model.pkl', 'rb') as f:
        nn_model = pickle.load(f)
    prompt_embeddings = np.load('prompt_embeddings.npy')

    user_seq = tokenizer.texts_to_sequences([normalize_text(user_text)])[0]
    user_embed = get_average_embedding(user_seq).reshape(1, -1)
    distances, indices = nn_model.kneighbors(user_embed)

    # Si hay alta similitud semántica, usar esa respuesta
    if distances[0][0] < 0.3:
        return completions_aug[indices[0][0]]

    # Manejar texto vacío
    if not user_seq:
        return "Lo siento, no te entendí."

    # Manejar mucho vocabulario desconocido
    oov_ratio = sum(1 for i in user_seq if i == oov_index) / len(user_seq)
    if oov_ratio > 0.4:
        return "No entendí bien. ¿Podrías decirlo de otra forma?"

    # Predecir con el modelo
    pad = pad_sequences([user_seq], maxlen=MAX_LEN, padding='post')
    pred = final_model.predict(pad, verbose=0)[0]

    # Analizar las mejores predicciones
    top_indices = pred.argsort()[-3:][::-1]
    top_responses = [distinct_responses[i] for i in top_indices]
    
    # Si hay ambigüedad en las predicciones
    if len(set(top_responses)) == 1:
        return "¿Podrías ser más específico para poder ayudarte mejor?"

    # Si la confianza es muy baja
    if np.max(pred) < 0.3:
        return "No estoy seguro de lo que quieres decir. ¿Podrías intentar reformular?"

    # Devolver la respuesta con mayor probabilidad
    idx = np.argmax(pred)
    return distinct_responses[idx]

def simulate_typing(text, delay=0.03):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# Interfaz de usuario
if __name__ == '__main__':
    memory = OrderedDict()
    print("Chatbot de Genética y Matemáticas listo. Escribe 'salir' para terminar.")
    while True:
        msg = input("Tú: ").strip().lower()
        if msg in ['salir', 'exit', 'quit']:
            print("Bot: ¡Hasta luego! Espero haberte ayudado con tus dudas.")
            break
        if msg in memory:
            respuesta = memory[msg]
        else:
            respuesta = generate_response(msg)
            if len(memory) >= 20:
                memory.popitem(last=False)
            memory[msg] = respuesta
        simulate_typing(f"Bot: {respuesta}", delay=0.03)