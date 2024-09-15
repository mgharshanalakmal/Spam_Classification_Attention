from model.utils import predict_word_impotance

from flask import Blueprint, render_template, request, jsonify
import logging
import warnings
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


views = Blueprint("views", __name__)


@views.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")


@views.route("/predict", methods=["POST"])
def predict():
    data = request.json
    email_text = data.get("email_text")
    prediction, highlighted_words = predict_word_impotance(email_text)
    print(prediction, highlighted_words)

    return jsonify({"prediction": prediction, "highlighted_words": highlighted_words})
