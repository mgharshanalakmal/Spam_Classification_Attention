from model.attention import Attention

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import os
import pickle
import keras
import logging
import tensorflow as tf


def predict_word_impotance(text):
    pred_results, word_dictionary = word_scale_vector(text)

    highlight_words = []

    cleaned_sentence = text.replace("\r", " ").replace("\n", " ")
    words = cleaned_sentence.split()

    for word in words:
        lower_word = word.lower().strip(",.!?;")
        if lower_word in word_dictionary:
            value = float(word_dictionary[lower_word])
            highlight_words.append({"word": word, "value": value})

        else:
            highlight_words.append({"word": word, "value": float(0)})

    return float(pred_results[0]), highlight_words


def word_scale_vector(text):
    pred_results, word_vector = predict__(text=text)

    df = pd.DataFrame(word_vector, columns=["Word", "Attention Weight"])
    scaler = MinMaxScaler()
    attention = scaler.fit_transform(df[["Attention Weight"]])

    attention = np.array(attention)
    words = np.array(df["Word"])

    word_dictionary = {key: value[0] for key, value in zip(words, attention)}

    return pred_results, word_dictionary


def predict__(text):
    saved_model_dictionary = load_saved_models(folder_path="model\saved_models")

    preprocessing_pipe = saved_model_dictionary["Preprocessing Pipe"]
    reverse_word_index = saved_model_dictionary["Reverse Word Index"]
    classification_model = saved_model_dictionary["Classification Model"]
    attention_model = saved_model_dictionary["Attention Model"]

    pred_text = np.array([text])
    processed_text = preprocessing_pipe.transform(pred_text)

    pred_results = classification_model.predict(processed_text)[0]

    attention_w = attention_model.predict(processed_text)[1][0]

    word_mat = []

    for vect, weight in zip(processed_text[0], attention_w):
        try:
            word = reverse_word_index[vect]
            word_mat.append((word, weight[0]))

        except KeyError:
            break

    return pred_results, word_mat


def load_saved_models(folder_path):
    preprocessing_pipe, reverse_word_index = load_preprocessing_models(folder_path)
    classification_model, attention_model = load_prediction_models(folder_path)

    model_dictionary = {
        "Preprocessing Pipe": preprocessing_pipe,
        "Reverse Word Index": reverse_word_index,
        "Classification Model": classification_model,
        "Attention Model": attention_model,
    }

    return model_dictionary


def load_preprocessing_models(file_path):
    preprocessing_file_name = "preprocessing.pkl"
    pickle_file = os.path.join(file_path, preprocessing_file_name)

    with open(pickle_file, "rb") as f:
        loaded_objects = pickle.load(f)

    preprocessing_pipe = loaded_objects["preprocessing_pipe"]
    reversed_word_index = loaded_objects["reverse_word_index"]

    return preprocessing_pipe, reversed_word_index


def load_prediction_models(file_path):
    tf.get_logger().setLevel(logging.ERROR)

    classification_model_name = "classification_model.keras"
    attention_model_name = "attention_model.keras"

    classification_file_path = os.path.join(file_path, classification_model_name)
    attention_file_path = os.path.join(file_path, attention_model_name)

    classification_model = keras.models.load_model(classification_file_path, custom_objects={"Attention": Attention})
    attention_model = keras.models.load_model(attention_file_path, custom_objects={"Attention": Attention})

    return classification_model, attention_model
