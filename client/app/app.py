import logging
logging.basicConfig(level=logging.DEBUG)

from flask import Flask, render_template, request, jsonify
import requests


app = Flask(
    __name__,
    template_folder="templates"
)


@app.route("/")
def index():

    logging.debug(f"A DEBUG Message from client.")

    return render_template("index.html")

@app.route("/run_pipeline", methods=["POST"])
def run_pipeline():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"message": "No data received"}), 400

        query = data.get("query")
        json_to_send = {
            "query": query
        }

        logging.debug(json_to_send)

        response = requests.post(f"http://query_retriever:8080/main_logic", json=json_to_send)
        response.raise_for_status()

        logging.debug(response)

        return response.json(), 200
    except Exception as e:
        return jsonify({"message": "Error processing request"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
