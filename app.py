import yaml
from flask import Flask, request, jsonify, render_template
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
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

# Create a custom prompt template with the system instruction
system_prompt_file = cfg['llm'].get('system_prompt', '')
system_prompt = open(system_prompt_file, 'r').read()
template = f"""{system_prompt}

Use the following pieces of content in the book (context) to answer the question at the end.
If you don't know the answer, just say that you don't know and don't try to make up an answer.

Context:
{{context}}

Question: {{question}}

Helpful Answer:"""

QA_PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)
COMBINE_DOCS_CHAIN_KWARGS = {"prompt": QA_PROMPT}


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

    chat_history = data.get("chat_history", [])
    titles = data.get("book_titles", [])
    language = data.get("language", "English")

    # Convert the chat history from a list of tuples to a list of Messages.
    formatted_chat_history = []
    for q, a in chat_history:
        formatted_chat_history.append(HumanMessage(content=q))
        formatted_chat_history.append(AIMessage(content=a))

    # Add language instruction to the question
    question_with_lang = f"{question}\n\n(Please provide the answer in {language})"

    search_kwargs = {
        "k": cfg['retrieval']['k'],
        "ef": cfg['retrieval']['ef'],
    }
    if titles:
        expr = f"book_title in {titles!r}"
        search_kwargs["expr"] = expr

    retriever = vs.as_retriever(search_kwargs=search_kwargs)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs=COMBINE_DOCS_CHAIN_KWARGS
    )

    res = qa.invoke({"question": question_with_lang, "chat_history": formatted_chat_history})
    answer = res["answer"]

    thinking_steps = []
    for doc in res["source_documents"]:
        thinking_steps.append({
            "book_title": doc.metadata.get("book_title", "Unknown"),
            "content": doc.page_content,
            "page": doc.metadata.get("page", "N/A")
        })

    chat_history.append((question, answer))

    return jsonify({
        "answer": answer,
        "sources": list({doc.metadata["book_title"] for doc in res["source_documents"]}),
        "chat_history": chat_history,
        "thinking": thinking_steps
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

