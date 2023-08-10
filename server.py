# Add these in requirements.txt
# sentence_transformers==2.2.2
# flask==2.3.2
# werkzeug==2.3.6

from flask import Flask, request
from werkzeug.utils import secure_filename

from privateGPT import PrivateGPT

app = Flask(__name__)

privateGPT = PrivateGPT()


@app.route("/ask", methods=['GET'])
def query():
  print("Query initialized")
  question = request.args.get("query")
  answer = privateGPT.process_query(question)
  return f'Question is {answer}'


@app.route("/upload", methods=['POST'])
def upload():
  print("File upload started")
  file = request.files['file']
  file.save(f"./source_documents/{secure_filename(file.filename)}")
  print("File upload complete")
  privateGPT.update_model()
  return "OK"
