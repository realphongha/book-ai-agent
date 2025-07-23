import yaml
from flask import Flask, request, jsonify, render_template
# Replace RetrievalQA with ConversationalRetrievalChain
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus

cfg = yaml.safe_load(open('configs.yaml', 'r'))

app = Flask(__name__)

# Load embeddings & vector store once at startup
embeddings = OllamaEmbeddings(model=cfg['embedding']['model'])
vs = Milvus(
    embedding_function=embeddings,
    connection_args={
        "host": cfg['milvus']['host'],
        "port": cfg['milvus']['port'],
    },
    collection_name=cfg['milvus']['collection'],
)
llm = OllamaLLM(model=cfg['llm']['model'])


@app.route('/')
def home():
    return render_template('chat.html')


@app.route('/api/books')
def books():
    res = vs.client.query(
        output_fields=["book_title"],
        collection_name=cfg['milvus']['collection'],
        filter='book_title != ""'
    )
    titles = sorted({r["book_title"] for r in res})
    return jsonify(titles)


@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    # Get chat history from the request, default to an empty list
    chat_history = data.get("chat_history", [])
    titles = data.get("book_titles", [])

    retr_kwargs = {"k": 3}
    if titles:
        expr = f"book_title in {titles!r}"
        retr_kwargs["search_kwargs"] = {"expr": expr}

    retriever = vs.as_retriever(**retr_kwargs)

    # Create the conversational chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Invoke the chain with the question and history
    res = qa.invoke({"question": question, "chat_history": chat_history})
    answer = res["answer"]

    # Append the new question and answer to the history
    chat_history.append((question, answer))

    return jsonify({
        "answer": answer,
        "sources": list({doc.metadata["book_title"] for doc in res["source_documents"]}),
        "chat_history": chat_history
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

