import re
import json
import string
import numpy as np
import unicodedata
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import time
import sys

# 🔧 AJUSTES PARA DATOS PEQUEÑOS/MEDIANOS
VOCAB_SIZE = 500
EMBEDDING_DIM = 16
MAX_LEN = 15
NUM_NEURONS = 64
EPOCHS = 90
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.5
THRESHOLD = 0.85  # Umbral para la red neuronal
VALIDATION_SPLIT = 0.3

# NORMALIZAR EL TEXTO
def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[^a-záéíóúñü¡!¿? ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# CARGAR Y PROCESAR DATOS DE ENTRENAMIENTO
with open("training.json", "r", encoding="utf-8") as f:
    data = json.load(f)

training_data = data["conversations"]
raw_prompts = [conv["prompt"] for conv in training_data]
prompts       = [normalize_text(p) for p in raw_prompts]
completions   = [conv["completion"].strip() for conv in training_data]
intents       = [conv["intent"] for conv in training_data]
prompt_token_sets = [set(p.split()) for p in prompts]

# TOKENIZACIÓN
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(prompts)
oov_index   = tokenizer.word_index[tokenizer.oov_token]
input_seqs  = tokenizer.texts_to_sequences(prompts)
padded_inputs = pad_sequences(input_seqs, maxlen=MAX_LEN, padding='post')

# CODIFICACIÓN DE RESPUESTAS
responses = sorted(list(set(completions)))
resp2idx  = {resp: i for i, resp in enumerate(responses)}
idx2resp  = {i: resp for resp, i in resp2idx.items()}
y_indices = np.array([resp2idx[c] for c in completions])
y         = to_categorical(y_indices, num_classes=len(responses))

# MODELO SECUENCIAL
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN, mask_zero=True),
    Bidirectional(LSTM(NUM_NEURONS, return_sequences=True)),
    Dropout(DROPOUT_RATE),
    Bidirectional(LSTM(NUM_NEURONS // 2)),
    Dense(NUM_NEURONS, activation='relu'),
    Dropout(DROPOUT_RATE),
    Dense(len(responses), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    "best_model.keras", save_best_only=True, monitor="val_loss", verbose=1
)

# ENTRENAMIENTO
try:
    history = model.fit(
        padded_inputs, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[checkpoint]
    )
except tf.errors.InternalError:
    print("WARNING: error interno de CUDA, continuando en CPU.")
    with tf.device('/CPU:0'):
        history = model.fit(
            padded_inputs, y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=[checkpoint]
        )

model.save("chatbot_model_final.keras")

# SIMILITUD Jaccard
def jaccard_similarity(set1, set2):
    inter = set1 & set2
    union = set1 | set2
    return len(inter) / len(union) if union else 0

# EXTRAER PALABRAS PRINCIPALES
def extract_keywords(text):
    norm = normalize_text(text)
    return set(norm.split())

# GENERAR RESPUESTA
def generate_response(input_text):
    norm = normalize_text(input_text)
    keyword_set = extract_keywords(norm)

    # Calcula similitudes Jaccard
    match_scores = [
        (i, jaccard_similarity(keyword_set, pts), pts & keyword_set)
        for i, pts in enumerate(prompt_token_sets)
    ]
    best_idx, best_sim, overlap = max(match_scores, key=lambda x: x[1])

    # 1) Sin límite: si comparte al menos 1 palabra, devuelve la respuesta
    if best_sim > 0:
        return completions[best_idx]

    # 2) Si no, fallback al modelo
    seq = tokenizer.texts_to_sequences([norm])[0]
    if oov_index in seq:
        return "Lo siento, no te entendí."
    pad = pad_sequences([seq], maxlen=MAX_LEN, padding='post')
    pred = model.predict(pad, verbose=0)[0]
    idx  = np.argmax(pred)
    if pred[idx] < THRESHOLD:
        return "Lo siento, no te entendí."
    return idx2resp[idx]

# SIMULACIÓN DE TYPING
def simulate_typing(text, delay=0.03):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# MODO INTERACTIVO
if __name__ == "__main__":
    print("Chatbot técnico (v2) listo. Escribí 'salir' para terminar.")
    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("Bot: ¡Chao!")
            break
        simulate_typing("Bot: " + generate_response(user_input))
