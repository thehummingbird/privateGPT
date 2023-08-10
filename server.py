from flask import Flask, request
from werkzeug.utils import secure_filename
import requests

from privateGPT import PrivateGPT

app = Flask(__name__)

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
def upload():
  print("URL download started")
  title = request.args.get('title')
  url = request.args.get('url')

  request = requests.get(url, allow_redirects=True)
  open(f"./source_documents/{title}", 'wb').write(request.content)

  print("URL file saved")
  return "OK"
