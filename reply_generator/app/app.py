import os
import logging
logging.basicConfig(level=logging.DEBUG)

from flask import Flask, request, jsonify

import requests


########################################
#                                      #
#  Тут скорее всего база не нужна,     #
#  но все зависимости для подключения  #
#  в контейнере присутствуют.          #
#                                      #
########################################


NAME = os.environ.get("DATABASE_USER")


app = Flask(__name__)


@app.route("/")
def index():

    logging.debug(f"A DEBUG Message from {NAME}.")

    return f"<h1>Hello from {NAME}!</h1>"


@app.route("/main_logic", methods=["POST"])
def main_logic():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"message": "No data received"}), 400

        query = data.get("query")
        
        ##  Ваша логика  ##

        query =  query + " " + NAME.upper()
        
        ##  Конец Вашей логики  ##
        
        logging.debug(query)

        return jsonify({"message": query}), 200
    except Exception as e:
        return jsonify({"message": "Error processing request"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
