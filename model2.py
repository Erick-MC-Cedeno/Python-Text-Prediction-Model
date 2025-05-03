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

# Carga modelo spaCy
nlp = spacy.load("es_core_news_sm")
memory = OrderedDict()

# CONFIGURACIÓN
VOCAB_SIZE       = 5000
EMBEDDING_DIM    = 100
MAX_LEN          = 60
NUM_NEURONS      = 32
EPOCHS           = 100
BATCH_SIZE       = 16
INITIAL_LR       = 2e-4
DROPOUT_RATE     = 0.5
L2_RATE          = 1e-4
JACCARD_THRESH   = 0.3
VALIDATION_SPLIT = 0.2
KFOLDS           = 5

# FUNCIONES AUXILIARES
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

# CARGA DATOS
try:
    with open('training.json', 'r', encoding='utf-8') as f:
        data = json.load(f)['conversations']
except Exception as e:
    print("Error al cargar el JSON:", e)
    sys.exit(1)

raw_prompts = [c['prompt'] for c in data]
raw_completions = [c['completion'].strip() for c in data]
prompts = [normalize_text(p) for p in raw_prompts]
prompt_sets = [set(p.split()) for p in prompts]
prompts_aug, completions_aug = augment_texts(prompts, raw_completions)

# TOKENIZACIÓN
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(prompts_aug)
oov_index = tokenizer.word_index[tokenizer.oov_token]
seqs = tokenizer.texts_to_sequences(prompts_aug)
padded_inputs = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')

# SALIDAS
distinct_responses = sorted(set(completions_aug))
resp2idx = {r: i for i, r in enumerate(distinct_responses)}
y_indices = np.array([resp2idx[c] for c in completions_aug])
y_onehot = to_categorical(y_indices, num_classes=len(distinct_responses))

# GUARDADO
tokenizer_filename = 'tokenizer.pkl'
model_filename = 'chatbot_model_final.keras'

with open(tokenizer_filename, 'wb') as f:
    pickle.dump(tokenizer, f)

with open('response_map.json', 'w') as f:
    json.dump(resp2idx, f)

# MODELO
def build_model():
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN, mask_zero=True),
        Bidirectional(LSTM(NUM_NEURONS, kernel_regularizer=l2(L2_RATE))),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        Dense(len(distinct_responses), activation='softmax', kernel_regularizer=l2(L2_RATE))
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

callbacks = [
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', verbose=1),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    LearningRateScheduler(warmup_scheduler)
]

# CROSS VALIDATION
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(padded_inputs):
    X_train, X_val = padded_inputs[train_idx], padded_inputs[val_idx]
    y_train, y_val = y_onehot[train_idx], y_onehot[val_idx]
    model = build_model()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

# ENTRENAMIENTO FINAL
final_model = build_model()
history_final = final_model.fit(
    padded_inputs, y_onehot,
    validation_split=VALIDATION_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2
)
final_model.save(model_filename)

# RESPUESTA

def load_tokenizer_and_responses():
    global tokenizer, distinct_responses, resp2idx
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('response_map.json', 'r') as f:
        resp2idx = json.load(f)
        distinct_responses = [None] * len(resp2idx)
        for r, i in resp2idx.items():
            distinct_responses[int(i)] = r

load_tokenizer_and_responses()

def reason_before_response(user_text):
    start_time = time.time()
    best_score = 0
    best_response = None
    kw = extract_keywords(user_text)

    for past_input, past_response in reversed(memory.items()):
        if time.time() - start_time > 5:
            break
        past_kw = extract_keywords(past_input)
        score = jaccard_similarity(kw, past_kw)
        if score > best_score:
            best_score = score
            best_response = past_response

    if best_score >= JACCARD_THRESH:
        return best_response

    for i, prompt_kw in enumerate(prompt_sets):
        if time.time() - start_time > 5:
            break
        score = jaccard_similarity(kw, prompt_kw)
        if score > best_score:
            best_score = score
            best_response = raw_completions[i]

    return best_response

def generate_response(user_text):
    response = reason_before_response(user_text)
    if response:
        return response

    seq = tokenizer.texts_to_sequences([normalize_text(user_text)])[0]
    if oov_index in seq:
        return "Lo siento, no te entendí."

    pad = pad_sequences([seq], maxlen=MAX_LEN, padding='post')
    pred = final_model.predict(pad, verbose=0)[0]

    if np.max(pred) < 0.3:
        return "No estoy seguro de lo que quieres decir. ¿Podrías intentar reformular?"

    idx = np.argmax(pred)
    return distinct_responses[idx]

def simulate_typing(text, delay=0.03):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(delay)
    print()

if __name__ == '__main__':
    memory = OrderedDict()
    print("Chatbot listo. Escribe 'salir' para terminar.")
    while True:
        msg = input("Tú: ").strip().lower()
        if msg in ['salir', 'exit', 'quit']:
            print("Bot: ¡Chao!")
            break

        if msg in memory:
            respuesta = memory[msg]
        else:
            respuesta = generate_response(msg)
            if len(memory) >= 20:
                memory.popitem(last=False)
            memory[msg] = respuesta

        simulate_typing(respuesta, delay=0.03)
