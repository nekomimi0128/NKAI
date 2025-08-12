from flask import Flask, request, jsonify
from model import generate_response  # さっき作ったモデルの関数

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "メッセージを送ってね"}), 400

    response = generate_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
