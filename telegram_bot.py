import telebot
from src import ChatBot
import pathlib

CURDIR = pathlib.Path(__file__).parent.resolve()
TOKENPATH = CURDIR / ".token"
assert TOKENPATH.exists()
with open(TOKENPATH, "r") as file:
    TOKEN = file.read().strip()
assert TOKEN

bot = telebot.TeleBot(TOKEN)
chatbot = ChatBot(use_vllm=True)


@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(message, "Hello, how are you doing?")


@bot.message_handler(commands=["help"])
def help(message):
    bot.reply_to(
        message, "I am a multiagent personalized chatbot. Just send me a message!"
    )


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_query = message.text
    try:
        reply = chatbot.response(user_query)
        bot.reply_to(message, reply)
    except Exception as e:
        bot.reply_to(message, "error")
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    bot.infinity_polling()
