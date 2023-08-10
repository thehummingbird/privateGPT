from flask import Flask, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
import requests

from privateGPT import PrivateGPT

app = Flask(__name__)
CORS(app)

privateGPT = PrivateGPT()

@app.route("/ask", methods=['GET'])
def query():
  print("Query initialized")
  question = request.args.get("query")
  answer = privateGPT.process_query(question)
  return answer


@app.route("/upload", methods=['POST'])
def upload():
  print("File upload started")
  file = request.files['file']
  file.save(f"./source_documents/{secure_filename(file.filename)}")
  print("File upload complete")
  privateGPT.update_model()
  return "OK"


@app.route("/feed", methods=['POST'])
def feed():
  print("URL download started")
  title = request.form.get('title')
  url = request.form.get('url')

  print(f"title: {title}, url: {url}")
  document = requests.get(url, allow_redirects=True)
  open(f"./source_documents/{title}", 'wb').write(document.content)
  privateGPT.update_model()
  print("URL file saved and ingested")
  return "OK"
