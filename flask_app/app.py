# app.py

import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates

app = Flask(__name__)
CORS(app)


# ---------------- PREPROCESS ---------------- #

def preprocess_comment(comment):
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {
            'not', 'but', 'however', 'no', 'yet'
        }

        comment = ' '.join(
            [word for word in comment.split() if word not in stop_words]
        )

        lemmatizer = WordNetLemmatizer()

        comment = ' '.join(
            [lemmatizer.lemmatize(word) for word in comment.split()]
        )

        return comment

    except Exception as e:
        print(e)
        return comment


# ---------------- LOAD MODEL ---------------- #

def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    mlflow.set_tracking_uri("http://13.51.166.199:5000")

    client = MlflowClient()

    model_uri = f"models:/{model_name}/{model_version}"

    model = mlflow.pyfunc.load_model(model_uri)

    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


model, vectorizer = load_model_and_vectorizer(
    "yt_chrome_plugin_model",
    "1",
    "./tfidf_vectorizer.pkl"
)


# ---------------- FEATURE ENGINEERING ---------------- #

def prepare_features(comments):

    preprocessed_comments = [
        preprocess_comment(comment)
        for comment in comments
    ]

    tfidf_features = vectorizer.transform(
        preprocessed_comments
    ).toarray()

    extra_features = []

    for comment in preprocessed_comments:

        words = comment.split()

        word_count = len(words)

        char_count = len(comment)

        avg_word_length = (
            sum(len(word) for word in words) / len(words)
            if len(words) > 0 else 0
        )

        extra_features.append([
            word_count,
            char_count,
            avg_word_length
        ])

    extra_features = np.array(extra_features)

    final_features = np.hstack(
        (tfidf_features, extra_features)
    )

    return final_features


# ---------------- HOME ---------------- #

@app.route('/')
def home():
    return "Welcome to our flask api"


# ---------------- PREDICT ---------------- #

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    comments = data.get('comments')

    if not comments:
        return jsonify({
            "error": "No comments provided"
        }), 400

    try:
        final_features = prepare_features(comments)

        predictions = model.predict(
            final_features
        ).tolist()

        label_map = {
            0: -1,   # Negative
            1: 0,    # Neutral
            2: 1     # Positive
        }

        predictions = [
            str(label_map[int(pred)])
            for pred in predictions
        ]

    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

    response = [
        {
            "comment": comment,
            "sentiment": sentiment
        }
        for comment, sentiment in zip(
            comments, predictions
        )
    ]

    return jsonify(response)


# ---------------- PREDICT WITH TIMESTAMP ---------------- #

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():

    data = request.json

    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({
            "error": "No comments provided"
        }), 400

    try:
        comments = [
            item['text']
            for item in comments_data
        ]

        timestamps = [
            item['timestamp']
            for item in comments_data
        ]

        final_features = prepare_features(comments)

        predictions = model.predict(
            final_features
        ).tolist()

        label_map = {
            0: -1,   # Negative
            1: 0,    # Neutral
            2: 1     # Positive
        }

        predictions = [
            str(label_map[int(pred)])
            for pred in predictions
        ]

    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

    response = [
        {
            "comment": comment,
            "sentiment": sentiment,
            "timestamp": timestamp
        }
        for comment, sentiment, timestamp in zip(
            comments, predictions, timestamps
        )
    ]

    return jsonify(response)


# ---------------- PIE CHART ---------------- #

@app.route('/generate_chart', methods=['POST'])
def generate_chart():

    try:
        data = request.get_json()

        sentiment_counts = data.get(
            'sentiment_counts'
        )

        labels = ['Positive', 'Neutral', 'Negative']

        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        plt.figure(figsize=(6, 6))

        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140
        )

        plt.axis('equal')

        img_io = io.BytesIO()

        plt.savefig(
            img_io,
            format='PNG',
            transparent=True
        )

        img_io.seek(0)

        plt.close()

        return send_file(
            img_io,
            mimetype='image/png'
        )

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ---------------- WORD CLOUD ---------------- #

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():

    try:
        data = request.get_json()

        comments = data.get('comments')

        preprocessed_comments = [
            preprocess_comment(comment)
            for comment in comments
        ]

        text = ' '.join(preprocessed_comments)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(
                stopwords.words('english')
            ),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()

        wordcloud.to_image().save(
            img_io,
            format='PNG'
        )

        img_io.seek(0)

        return send_file(
            img_io,
            mimetype='image/png'
        )

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ---------------- TREND GRAPH ---------------- #

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():

    try:
        data = request.get_json()

        sentiment_data = data.get(
            'sentiment_data'
        )

        df = pd.DataFrame(sentiment_data)

        df['timestamp'] = pd.to_datetime(
            df['timestamp']
        )

        df.set_index(
            'timestamp',
            inplace=True
        )

        df['sentiment'] = df[
            'sentiment'
        ].astype(int)

        monthly_counts = df.resample(
            'M'
        )['sentiment'].value_counts().unstack(
            fill_value=0
        )

        monthly_totals = monthly_counts.sum(
            axis=1
        )

        monthly_percentages = (
            monthly_counts.T / monthly_totals
        ).T * 100

        for col in [-1, 0, 1]:
            if col not in monthly_percentages.columns:
                monthly_percentages[col] = 0

        monthly_percentages = monthly_percentages[
            [-1, 0, 1]
        ]

        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',
            0: 'gray',
            1: 'green'
        }

        labels = {
            -1: 'Negative',
            0: 'Neutral',
            1: 'Positive'
        }

        for val in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[val],
                marker='o',
                label=labels[val],
                color=colors[val]
            )

        plt.xticks(rotation=45)

        plt.tight_layout()

        img_io = io.BytesIO()

        plt.savefig(img_io, format='PNG')

        img_io.seek(0)

        plt.close()

        return send_file(
            img_io,
            mimetype='image/png'
        )

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
        
        
@app.route('/process_comments', methods=['POST'])
def process_comments():
    data = request.json
    comments = data.get("comments", [])

    joined = " ".join([c["text"] for c in comments[:100]])

    return jsonify({
        "processed_comments": joined
    })


# ---------------- RUN ---------------- #

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False
    )