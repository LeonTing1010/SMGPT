#!/usr/bin/env python3
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import argparse
import csv
import os
from langchain.llms import GPT4All, LlamaCpp
from langchain.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from constants import CHROMA_SETTINGS
from dotenv import load_dotenv
load_dotenv()


embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')


def main():
    # Parse the command line arguments
    args = parse_arguments()
    # embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM

    if model_type == "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=True, n_threads=16, repeat_penalty=2.1, temperature=0)
    elif model_type == "GPT4All":
        llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
    else:
        llm = ChatOpenAI(temperature=0, callbacks=callbacks, verbose=True)
        print(f"Model {model_type} not supported!")
        exit

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)
    # Interactive questions and answers
    with open("SMGPT/qa.csv", 'a') as f:
        writer = csv.writer(f)

        while True:
            query = input("\nEnter a query: ")
            if query == "exit":
                break

            # Get the answer from the chain
            res = qa(query)
            answer = res['result']

            # Write to CSV
            writer.writerow([query, answer])
            # Get the answer from the chain
            res = qa(query)
            answer, docs = res['result'], [] if args.hide_source else res['source_documents']

            # Print the result
            print("\n\n> Question:")
            print(query)
            print("\n> Answer:")
            print(answer)

            # Print the relevant sources used for the answer
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)


def parse_arguments():
    parser = argparse.ArgumentParser(description='SMGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
