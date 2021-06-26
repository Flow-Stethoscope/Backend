import os

os.system("apt install libsndfile1")

import json

import pymongo
from flask import Flask, request, redirect
import bcrypt
from api import firebase

from inference import Inference
from schemas import User

import ast


client = pymongo.MongoClient(
    "mongodb+srv://admin:mAsSeYhAcKeZ2096@flow-db.wcavd.mongodb.net/test"
)
db = client["flow"]
inference = Inference()

# files = request.files


ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif", "mp3"}
app = Flask(__name__)
app.run()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def hello_world():
    return {"Health": "Good"}


@app.route("/test", methods=["GET"])
def test():
    return redirect("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


@app.route("/register_user", methods=["POST"])
def register_user():
    db_user = request.form.to_dict()
    if User.user.is_valid(db_user):
        if (db["users"].find_one({"username": db_user["username"]})) is not None:
            return "This username already exists"
        db_user["password"] = bcrypt.hashpw(
            db_user["password"].encode("utf-8"), bcrypt.gensalt()
        )
        db_user["recordings"] = []
        if db_user["user_type"] == "Patient":
            db_user["doctors"] = []
            db["patients"].insert_one(db_user)
        else:
            db_user["patients"] = []
            db["doctors"].insert_one(db_user)
        db["users"].insert_one(db_user)
        return "Success"
    return "Form data incorrectly formatted"


@app.route("/login/<username>", methods=["POST"])
def login(username):
    return str(
        bcrypt.checkpw(
            request.form.to_dict()["password"].encode("utf-8"),
            db["users"].find_one({"username": username})["password"],
        )
    )


@app.route("/users/<username>", methods=["GET"])
def user(username):
    user_info = db["users"].find_one({"username": username})
    user_info["_id"] = None
    user_info["password"] = None
    print(user_info)
    return json.loads(json.dumps(user_info))


# client sends bytes to backend
@app.route("/send_recording", methods=["POST"])
def send_recording():
    recording = request.form.to_dict()

    print("recording from route:", recording)

    bytes = recording["recording"]

    if isinstance(bytes, str):
        print("converting")
        bytes = ast.literal_eval(bytes)
    if isinstance(bytes, int):
        bytes = [bytes]

    bytes = [float(byte) for byte in bytes]

    print("bytes type", type(bytes))
    print("bytes", bytes)

    classification = inference.predict(bytes)
    # path = inference.get_wav(bytes)
    # url = firebase.uploadFile(path)
    # inference.delete_wav(path)
    print("deleted file")
    rec_id = (
        db["recordings"]
        .insert_one(
            {
                "url": "no mask L",
                "username": recording["username"],
                "classification": classification,
            }
        )
        .inserted_id
    )
    print("rec_id found")
    db.users.update(
        {"username": recording["username"]},
        {
            "$push": {
                "recordings": str(db["recordings"].find_one({"_id": rec_id}))
            }
        },
    )
    db.patients.update(
        {"username": recording["username"]},
        {
            "$push": {
                "recordings": str(db["recordings"].find_one({"_id": rec_id}))
            }
        },
    )
    print("db updated")
    return {"recording_id": str(rec_id)}

