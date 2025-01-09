from time import sleep
import os
import logging
logging.basicConfig(level=logging.DEBUG)

from flask import Flask, request, jsonify
import mysql.connector

import requests


NAME = os.environ.get("DATABASE_USER")


class DbManager:

    def __init__(self, timeout=15) -> None:

        db_host     = os.environ.get("DATABASE_HOST")
        db_user     = os.environ.get("DATABASE_USER")
        db_password = os.environ.get("DATABASE_PASSWORD")
        db_database = os.environ.get("DATABASE_NAME")
        db_port     = 3306

        for _ in range(timeout):
            try:
                self.connection = mysql.connector.connect(
                    host=db_host,
                    user=db_user,
                    password=db_password,
                    port=db_port,
                    database=db_database,
                )
                break
            except Exception as e:
                print(e)
                print("Wainting for database to start up...")
                sleep(1)
        else:
            raise TimeoutError(f"Sth went wrong on {self.__class__.__name__} startup.")
        
        self.cursor = self.connection.cursor()
        print("Connected successfully!")

    def __del__(self):
        self.connection.close()


dbmanager = DbManager()
app       = Flask(__name__)


@app.route("/")
def index():

    # dbmanager.cursor.execute(
    #     """
    #         SHOW TABLES;
    #     """
    # )
    # tables = dbmanager.cursor.fetchall()

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

        json_to_send = {
            "query": query
        }

        logging.debug(json_to_send)

        response = requests.post(f"http://db_retriever:8080/main_logic", json=json_to_send)

        return response.json(), 200
    except Exception as e:
        return jsonify({"message": "Error processing request"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
