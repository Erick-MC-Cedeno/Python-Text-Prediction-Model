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
from sklearn.feature_extraction.text import CountVectorizer
import time 
import sys

# 🔧 AJUSTES PARA DATOS PEQUEÑOS/MEDIANOS
VOCAB_SIZE = 500            # ↓↓↓ Vocabulario ajustado al tamaño real de palabras únicas
EMBEDDING_DIM = 16         # ↓↓ 75% menos dimensión que el original
MAX_LEN = 15                # ↓↓ Reduce a la mitad (analiza longitud promedio de tus prompts)
NUM_NEURONS = 64             # ↓↓ Modelo 4x más pequeño
EPOCHS = 90                # ↓ Mitad de épocas (early stopping actuará antes)
BATCH_SIZE = 4              # ↓ Mitad de tamaño (mejor para generalización)
LEARNING_RATE = 1e-4        # ↓↓↓ Tasa ultra conservadora
DROPOUT_RATE = 0.5            # ↑↑↑ Regularización agresiva (70% de dropout)
THRESHOLD = 0.85            # ↑↑ Mayor exigencia para predicciones
SEGMENT_THRESHOLD = 0.5     # ↑ Más rigor en coincidencia por keywords
VALIDATION_SPLIT = 0.3      # ↑ 30% para validación

# Hiperparámetros
# VOCAB_SIZE = 3000  
# EMBEDDING_DIM = 128
# MAX_LEN = 60
# NUM_NEURONS = 256
# EPOCHS = 70
# BATCH_SIZE = 10
# THRESHOLD = 0.65
# SEGMENT_THRESHOLD = 0.4

# NORMALIZAR EL TEXTO
def normalize_text(text):
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# CARGAR Y PROCESAR DATOS DE ENTRENAMIENTO
with open("training.json", "r", encoding="utf-8") as f:
    data = json.load(f)

training_data = data["conversations"]

raw_prompts = [conv["prompt"] for conv in training_data]
prompts = [normalize_text(p) for p in raw_prompts]
completions = [conv["completion"].strip() for conv in training_data]
intents = [conv["intent"] for conv in training_data]
prompt_token_sets = [set(p.split()) for p in prompts]


# TOKENIZACION
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(prompts)
oov_index = tokenizer.word_index[tokenizer.oov_token]
input_seqs = tokenizer.texts_to_sequences(prompts)
padded_inputs = pad_sequences(input_seqs, maxlen=MAX_LEN, padding='post')

# CODIFICACION DE RESPUESTAS
responses = sorted(list(set(completions)))
resp2idx = {resp: i for i, resp in enumerate(responses)}
idx2resp = {i: resp for resp, i in resp2idx.items()}
y_indices = np.array([resp2idx[c] for c in completions])
y = to_categorical(y_indices, num_classes=len(responses))

# MODELO SECUENCIAL
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN, mask_zero=True),
    Bidirectional(LSTM(NUM_NEURONS, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(NUM_NEURONS // 2)),
    Dense(NUM_NEURONS, activation='relu'),
    Dropout(0.3),
    Dense(len(responses), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    "best_model.keras", save_best_only=True, monitor="val_loss", verbose=1
)

# ENTRENAMIENTO
try:
    history = model.fit(
        padded_inputs, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[checkpoint]
    )
except tf.errors.InternalError:
    print("WARNING: error interno de CUDA, continuando en CPU.")
    with tf.device('/CPU:0'):
        history = model.fit(
            padded_inputs, y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            callbacks=[checkpoint]
        )

model.save("chatbot_model_final.keras")

# SIMILITUD Jaccard
def jaccard_similarity(set1, set2):
    inter = set1 & set2
    union = set1 | set2
    return len(inter) / len(union) if union else 0

# FUNCION PARA EXTRAER PALABRAS PRINCIPALES
def extract_keywords(text, top_k=100):
    norm = normalize_text(text)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([norm])
    words = vectorizer.get_feature_names_out()
    return set(words[:top_k]) if len(words) >= 2 else set(words)

# GENERAR UNA RESPUESTA
def generate_response(input_text):
    norm = normalize_text(input_text)

    # COINCIDENCIA DE PALABRAS
    keyword_set = extract_keywords(norm)
    match_scores = [
        (i, jaccard_similarity(keyword_set, pts), pts & keyword_set)
        for i, pts in enumerate(prompt_token_sets)
    ]

    # BUSCA LA MEJOR COINCIDENCIA DE AL MENOS 2 PALABRAS
    best_idx, best_sim, overlap = max(match_scores, key=lambda x: x[1])
    if len(overlap) >= 2 and best_sim >= SEGMENT_THRESHOLD:
        return completions[best_idx]

    # SI NO IR A EL MODELO
    seq = tokenizer.texts_to_sequences([norm])[0]
    if oov_index in seq:
        return "Lo siento, no te entendí."
    pad = pad_sequences([seq], maxlen=MAX_LEN, padding='post')
    pred = model.predict(pad, verbose=0)[0]
    idx = np.argmax(pred)
    if pred[idx] < THRESHOLD:
        return "Lo siento, no te entendí."
    return idx2resp[idx]

# SIMULACION DE RESPUESTA DE AI CON DELAY
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