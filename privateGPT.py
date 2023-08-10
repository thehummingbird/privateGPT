#!/usr/bin/env python3
from constants import CHROMA_SETTINGS
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import time
from ingest import ingest
import json
load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))


class PrivateGPT(object):
  def __init__(self):
    self.llm = None
    # self.args = None
    self.update_model()

  def update_model(self):

    print(f"Ingesting Data")
    ingest()

    print(f"Updating the model")

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [StreamingStdOutCallbackHandler()]

    # Prepare the LLM
    match model_type:
      case "LlamaCpp":
        self.llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx,
                            n_batch=model_n_batch, callbacks=callbacks, verbose=False)
      case "GPT4All":
        self.llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj',
                           n_batch=model_n_batch, callbacks=callbacks, verbose=False)
      case _default:
        # raise exception if model_type is not supported
        raise Exception(
            f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    self.qa = RetrievalQA.from_chain_type(
        llm=self.llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    print(f"Model update complete")

  def process_query(self, query):

    # Get the answer from the chain
    start = time.time()
    res = self.qa(query)
    answer, docs = res['result'], res['source_documents']
    reference = docs[0].metadata["source"].split("/")[-1]
    end = time.time()

    # Print the result
    print("\n\n> Question:")
    print(query)
    print(f"\n> Answer (took {round(end - start, 2)} s.):")
    print(answer)
    result = {
        "answer": answer,
        "reference": reference
    }
    return result


if __name__ == "__main__":
  privateGPT = PrivateGPT()
  while True:
    query = input("\nEnter a query: ")
    privateGPT.process_query(query)
