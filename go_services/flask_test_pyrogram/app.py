from flask import Flask, request
from flask import jsonify
import os
from pyrogram import Client, MessageHandler


app = Flask(__name__)

last_msg = "default"
broken = False


tg_client = Client("mos_ai_test",
                   config_file=os.path.join(os.path.split(app.instance_path)[0], 'config.ini'),
                   workdir=os.path.join(os.path.split(app.instance_path)[0]))



@tg_client.on_message()
def my_handler(client, message):
    global last_msg
    global broken
    if message.media:
        print("GOT MEDIA")

        try:
            last_msg = "started downloading photo"
            client.download_media(message,
              file_name=os.path.join(os.path.split(app.instance_path)[0], "pics/"))
            last_msg = "photo"
        except Exception as e:
            last_msg = "error with downloading file: " + str(e)
    else:
        print("GOT FROM BOT:" + message.text)
        last_msg = message.text

tg_client.start()

@app.route('/', methods=["GET"])
def index():
    print("LAST MSG: ", last_msg)
    return jsonify({"message": last_msg})


@app.route('/send', methods=['POST'])
def say_hello():
    content = request.json
    token = content['token']
    tg_client.send_message("mos_med_ai_bot", token)
    print("Started")
    print("POST ", token)
    return token

@app.route('/exit', methods=['GET'])
def stop():
    print("Stopping server")
    os._exit(0)

