import pandas as pd
import numpy as np
import os
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

BASE_PATH = "/kaggle/input/datasets/organizations/nih-chest-xrays/data" 
CSV_PATH = os.path.join(BASE_PATH, "Data_Entry_2017.csv")
IMG_SIZE = 224
BATCH_SIZE = 32

df = pd.read_csv(CSV_PATH)

# Izdvajanje labela i multi-hot encoding
all_labels = (
    df["Finding Labels"]
    .str.split("|")
    .explode()
    .unique()
)
all_labels = sorted([label for label in all_labels if label != "No Finding"])  

for label in all_labels:
    df[label] = df["Finding Labels"].apply(lambda x: 1 if label in x else 0)

# Mapiranje putanja do slika
image_paths = {}
for root, dirs, files in os.walk(BASE_PATH):
    for file in files:
        if file.endswith(".png"):
            image_paths[file] = os.path.join(root, file)

df["path"] = df["Image Index"].map(image_paths)
df = df.dropna(subset=["path"])

# Podjela na Train (70%), Val (15%), Test (15%)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# WEIGHTED LOSS FUNKCIJA
pos_weights = []
for label in all_labels:
    pos = train_df[label].sum()
    neg = len(train_df) - pos
    pos_weights.append(neg / (pos + 1e-5))

pos_weights_tensor = tf.constant(np.array(pos_weights), dtype=tf.float32)

def weighted_binary_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    loss = - (pos_weights_tensor * y_true * tf.math.log(y_pred) +
              (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

# GENERATORI SLIKA
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="path",
    y_col=all_labels,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="raw"
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="path",
    y_col=all_labels,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="raw"
)

# DEFINISANJE MODELA (Transfer Learning)
base_model = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(len(all_labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# CALLBACKS
callbacks_list = [
    ModelCheckpoint("best_model.h5", monitor="val_auc", mode="max", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_auc", mode="max", patience=3, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2, min_lr=1e-7, verbose=1)
]

# TRENING - FAZA 1: Frozen Backbone ---
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=weighted_binary_crossentropy,
    metrics=[tf.keras.metrics.AUC(multi_label=True, name="auc"), "binary_accuracy"]
)

history_frozen = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=7,
    callbacks=callbacks_list
)

# TRENING - FAZA 2: Fine-tuning
for layer in base_model.layers[-60:]:
    layer.trainable = True
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=weighted_binary_crossentropy,
    metrics=[tf.keras.metrics.AUC(multi_label=True, name="auc"), "binary_accuracy"]
)

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=8,
    callbacks=callbacks_list
)

# EVALUACIJA NA TEST SKUPU
test_generator = val_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="path",
    y_col=all_labels,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False
)

results = model.evaluate(test_generator)