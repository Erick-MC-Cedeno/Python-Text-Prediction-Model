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
from collections import OrderedDict
memory = OrderedDict()


# CONFIGURACIÓN DE PARAMETROS
VOCAB_SIZE       = 310       # Sin cambios, este valor está bien para un vocabulario pequeño.
EMBEDDING_DIM    = 100       # Mantener en 100, un tamaño estándar y adecuado.
MAX_LEN          = 60        # Aumentamos MAX_LEN a 40 para cubrir frases más largas (ajustar según la longitud máxima de tus frases).
NUM_NEURONS      = 32        # Mantener el número de neuronas en la capa LSTM.
EPOCHS           = 100       # Reducido a 100 para evitar sobreajuste.
BATCH_SIZE       = 16        # Tamaño de lote aumentado a 16 para mayor eficiencia.
INITIAL_LR       = 2e-4      # Ajuste del learning rate a 2e-4.
DROPOUT_RATE     = 0.5       # Mantener el dropout rate en 0.5.
L2_RATE          = 1e-4      # Regularización L2 en 1e-4.
JACCARD_THRESH   = 0.3       # Umbral de similitud Jaccard aumentado a 0.3.
VALIDATION_SPLIT = 0.2       # Mantener 20% para la validación.
KFOLDS           = 5         # Validación cruzada con 5 particiones.



# FUNCIONES AUXILIARES ---
def warmup_scheduler(epoch, lr):
    # LR Warm-up: incrementar gradualmente durante primeras 5 épocas
    if epoch < 5:
        return lr + (INITIAL_LR - 1e-5) / 5
    return lr


# NORMALIZAR TEXTO
def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[^a-záéíóúñü¡!¿? ]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


# AUGMENTAR DATOS
def augment_texts(texts, completions):
    # Ejemplo: duplicar dataset (puedes reemplazar con back-translation o sinónimos)
    return texts + texts, completions + completions


# CALCULAR JACCARD
def jaccard_similarity(a, b):
    inter = a & b
    uni = a | b
    return len(inter) / len(uni) if uni else 0


# EXTRAER PALABRAS CLAVE
def extract_keywords(text):
    return set(normalize_text(text).split())



# CARGAR FASTTEXT (OPCIONAL)
def load_fasttext(path, word_index, dim):
    matrix = np.zeros((VOCAB_SIZE, dim))
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            vals = line.rstrip().split(' ')
            word = vals[0]
            if word in word_index and word_index[word] < VOCAB_SIZE:
                matrix[word_index[word]] = np.array(vals[1:], dtype='float32')
    return matrix



#  CARGA Y PROCESO DE DATOS ---
with open('training.json', 'r', encoding='utf-8') as f:
    data = json.load(f)['conversations']
raw_prompts = [c['prompt'] for c in data]
raw_completions = [c['completion'].strip() for c in data]
prompts = [normalize_text(p) for p in raw_prompts]
prompt_sets = [set(p.split()) for p in prompts]

# DATA AUGMENTACIÓN
prompts_aug, completions_aug = augment_texts(prompts, raw_completions)


# TOKENIZACIÓN
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(prompts_aug)
oov_index = tokenizer.word_index[tokenizer.oov_token]
seqs = tokenizer.texts_to_sequences(prompts_aug)
padded_inputs = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')


# CODIFICACIÓN DE SALIDAS
distinct_responses = sorted(set(completions_aug))
resp2idx = {r:i for i,r in enumerate(distinct_responses)}
y_indices = np.array([resp2idx[c] for c in completions_aug])
y_onehot = to_categorical(y_indices, num_classes=len(distinct_responses))


# CARGA EMBEDDINGS SI LO DESEAS
# EMBEDDING_MATRIX = load_fasttext('cc.es.300.vec', tokenizer.word_index, EMBEDDING_DIM)
embedding_matrix = None


# CONSTRUYE EL MODELO
def build_model(embedding_matrix=None):
    layers = []
    if embedding_matrix is not None:
        layers.append(Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_LEN,
                                trainable=False))
    else:
        layers.append(Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                                input_length=MAX_LEN,
                                mask_zero=True))
    layers += [Bidirectional(LSTM(NUM_NEURONS, kernel_regularizer=l2(L2_RATE))),
               BatchNormalization(),
               Dropout(DROPOUT_RATE),
               Dense(len(distinct_responses), activation='softmax',
                     kernel_regularizer=l2(L2_RATE))]
    m = Sequential(layers)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return m

# CALLBACKS COMUNES
callbacks = [
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', verbose=1),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    LearningRateScheduler(warmup_scheduler)
]

# CROSS-VALIDACIÓN DE TRAINING
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
histories = []
for train_idx, val_idx in kf.split(padded_inputs):
    X_train, X_val = padded_inputs[train_idx], padded_inputs[val_idx]
    y_train, y_val = y_onehot[train_idx], y_onehot[val_idx]
    model = build_model(embedding_matrix)
    h = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )
    histories.append(h)


# ENTRENAMIENTO FINAL CON VALIDACIÓN
final_model = build_model(embedding_matrix)
history_final = final_model.fit(
    padded_inputs, y_onehot,
    validation_split=VALIDATION_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2
)
final_model.save('chatbot_model_final.keras')


# GENERACIÓN DE RESPUESTA Y CONSOLA
def generate_response(user_text):
    # Fallback por Jaccard
    kw = extract_keywords(user_text)
    scores = [(i, jaccard_similarity(kw, s)) for i, s in enumerate(prompt_sets)]
    best_i, best_sim = max(scores, key=lambda x: x[1])
    if best_sim > JACCARD_THRESH:
        return raw_completions[best_i]


    # MODELO LSTM
    seq = tokenizer.texts_to_sequences([normalize_text(user_text)])[0]
    if oov_index in seq:
        return "Lo siento, no te entendí."
    pad = pad_sequences([seq], maxlen=MAX_LEN, padding='post')
    pred = final_model.predict(pad, verbose=0)[0]
    idx = np.argmax(pred)
    return distinct_responses[idx] if pred[idx] >= 0.3 else "Lo siento, no te entendí."


# SIMULACIÓN DE RESPUESTA DE AI CON DELAY
def simulate_typing(text, delay=0.03):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(delay)
    print()


# MAIN Y MEMORIA TEMPORAL
if __name__ == '__main__':
    memory = OrderedDict()
    print("Chatbot listo. Escribe 'salir' para terminar.")
    while True:
        msg = input("Tú: ").strip().lower()
        if msg in ['salir', 'exit', 'quit']:
            print("Bot: ¡Chao!")
            break

        # MEMORIA TEMPORAL
        if msg in memory:
            respuesta = memory[msg]
        else:
            respuesta = generate_response(msg)
            if len(memory) >= 20:
                memory.popitem(last=False) 
            memory[msg] = respuesta

        simulate_typing(respuesta, delay=0.03)
