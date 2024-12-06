from flask import Flask, request, jsonify
from llm_model.model import load_model, scorer
import os
app = Flask(__name__)

# Load the LLM model
model = load_model()

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "Text not provided"}), 400

    text = [data["text"]]

    try:
        # Get the sentiment score using your pre-written function
        score = scorer(model, text)
        return jsonify({"text": text, "sentiment_score": score})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health_check():
    return "API is running", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get(PORT)))
