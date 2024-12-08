import os
import time
from dotenv import load_dotenv
import streamlit as st
from typing import Generator, List, Optional, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from pinecone import Pinecone, ServerlessSpec
from typing_extensions import TypedDict

load_dotenv()

tavily_api_key = st.secrets["api_keys"]["tavily"]
langsmith_api_key = st.secrets["api_keys"]["langsmith"]
pinecone_api_key = st.secrets["api_keys"]["pinecone"]
groq_api_key = st.secrets["api_keys"]["groq"]

pc = Pinecone(api_key=pinecone_api_key)

index_name = "langchain-test-index"  # Nombre del índice

existing_indexes = [index_info.name for index_info in pc.list_indexes()] # Corregido
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status.ready: # Corregido
        time.sleep(1)

index = pc.Index(index_name)

# Crear embeddings usando el modelo nomic-embed-text-v1.5
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={"trust_remote_code": True}
)

# Reconstruir el vector store con el índice existente
vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text") # Añadido text_key

# Configurar el retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})



from typing import List, Optional
from typing_extensions import TypedDict


class GraphState(TypedDict):

    conversation_history: List[str]
    question: Optional[str]
    documents: List[str]
    summarized_history: Optional[str]
    final_response: Optional[str]


#nodos

def retrieve(state):

    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    retrieved_docs = retriever.invoke(question)
    documents = [
        {"doc_id": doc.id, "content": doc.page_content} for doc in retrieved_docs
    ]
    
    return {"documents": documents, "question": question}

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuración del modelo y plantilla de prompts
llm = ChatGroq(temperature=0, model_name="llama-3.2-3b-preview", api_key=groq_api_key)

system_template = """
You are a grader assessing the relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Provide a binary score: 'yes' if the document is relevant, or 'no' if it is not.
Always output:
  "binary_score": "yes" | "no"
"""
prompt = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "Question: {question}\n\nDocument: {document}")]
)

# Crear el chain
chain_relevance = prompt | llm | StrOutputParser()

# Nodo para evaluar relevancia
def evaluate_relevance(state):
    """
    Nodo para evaluar la relevancia de los documentos recuperados.
    """
    print("---EVALUATE RELEVANCE---")
    # Validar que existan preguntas y documentos
    if not state.get("question") or not state.get("documents"):
        raise ValueError("El estado debe contener una pregunta y documentos recuperados.")
    
    

    # Evaluar cada documento
    for doc in state["documents"]:
        prompt_input = {
            "question": state["question"],
            "document": doc["content"]
        }
        try:
            # Invocar el chain
            response = chain_relevance.invoke(prompt_input)
            print("--RESPONSE--")
            print(response)
            print("------------")
            # Asignar relevancia según la respuesta del modelo
            doc["relevance"] = "yes" if "yes" in response else "no"
        except Exception as e:
            print(f"Error evaluando el documento {doc['doc_id']}: {e}")
            doc["relevance"] = "no"

    return state


def router_gen(state):

    print("---ROUTER---")
    # Verificar si hay documentos relevantes
    relevant_docs = [doc for doc in state["documents"] if doc.get("relevance") == "yes"]
    
    # Si no hay documentos relevantes
    if not relevant_docs:
        return "web_search"
    
    # Si hay documentos relevantes
    return "final_rag"

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=2)


def web_search(state):
    """
    Nodo para realizar una búsqueda en la web basada en la pregunta actual.
    Los resultados se agregan como documentos al estado con "relevance": "yes".

    Args:
        state (dict): El estado actual del flujo.

    Returns:
        dict: Estado actualizado con resultados de la web agregados.
    """
    print("--- WEB SEARCH NODE ---")
    
    # Validar que exista una pregunta en el estado
    if "question" not in state or not state["question"]:
        raise ValueError("El estado debe contener una pregunta válida en 'question'.")

    # Recuperar la pregunta actual
    question = state["question"]

    # Realizar la búsqueda web
    try:
        search_results = web_search_tool.invoke({"query": question})
    except Exception as e:
        print(f"Error en la búsqueda web: {e}")
        return state  # Devuelve el estado sin cambios si falla la búsqueda

    # Procesar los resultados de búsqueda
    web_documents = [
    {
        "doc_id": f"web_{i}",
        "content": result["content"],  # Aquí ya no se asume que result es un diccionario
        "metadata": {"source": "web_search"},
        "relevance": "yes"
    }
    for i, result in enumerate(search_results)
]
    # Agregar los resultados al estado
    if "documents" not in state:
        state["documents"] = []
    state["documents"].extend(web_documents)

    # Retornar el estado actualizado
    return state


# Configuración de `chain_rag`
llm= ChatGroq(temperature=0, model_name="llama-3.2-3b-preview", api_key=groq_api_key)

rag_template = """
You are an assistant for question-answering tasks.

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this question using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:
"""
rag_prompt = ChatPromptTemplate.from_messages(
    [("system", rag_template), ("user", "Context: {context}\n\nQuestion: {question}")]
)

# Crear el chain para `rag`
chain_rag = rag_prompt | llm | StrOutputParser()


def final_rag(state):
    """
    Genera una respuesta final utilizando los documentos relevantes en el estado.

    Args:
        state (dict): Estado actual con documentos recuperados, relevancias y pregunta.

    Returns:
        dict: Estado actualizado con la respuesta final.
    """
    print("--- FINAL RAG NODE ---")
    # Filtrar documentos relevantes
    relevant_docs = [doc for doc in state["documents"] if doc.get("relevance") == "yes"]

    # Generar respuesta si hay documentos relevantes
    if relevant_docs:
        context = " ".join(doc["content"] for doc in relevant_docs)
        response = chain_rag.invoke({"context": context, "question": state["question"]})
    else:
        # Mensaje predeterminado si no hay documentos relevantes
        response = "No se encontraron documentos relevantes. Considere realizar una búsqueda web."

    # Actualizar el estado con la respuesta final
    state["final_response"] = response

    # Retornar el estado completo
    return state

#grafo

from langgraph.graph import END, StateGraph, START

# Crear el workflow
workflow = StateGraph(GraphState)

# Definir los nodos
workflow.add_node("retrieve", retrieve)  # Nodo para recuperar documentos
workflow.add_node("evaluate_relevance", evaluate_relevance)  # Nodo para evaluar relevancia
workflow.add_node("web_search", web_search)  # Nodo para búsqueda en la web
workflow.add_node("final_rag", final_rag)  # Nodo para respuesta final

# Construir el grafo
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "evaluate_relevance")
workflow.add_conditional_edges(
    "evaluate_relevance",
    router_gen,
    {
        "web_search": "web_search",
        "final_rag": "final_rag",
    },
)

workflow.add_edge("web_search", "final_rag")  # Conexión desde web_search a final_rag
workflow.add_edge("final_rag", END)  # Finalizar flujo

# Compilar el workflow
app = workflow.compile()

# --- Interfaz de Streamlit ---

st.title("🤖 Física Molecular Bot ")

def generate_chat_responses(response_text: str) -> Generator[str, None, None]:
    for char in response_text:
        yield char

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("¿Qué quieres saber?")

if prompt:
     st.chat_message("user").markdown(prompt)
     st.session_state.messages.append({"role": "user", "content": prompt})

     try:
         with st.spinner("✨ Generando respuesta..."):
             inputs = {"question": prompt, "convesrsation_history": []} # Se agrega historial de conversacion.
             result = app.invoke(inputs)
             final_response = result.get("final_response")

             if final_response:
                 with st.chat_message("assistant"):
                     st.markdown(final_response)
                     st.session_state.messages.append({"role": "assistant", "content": final_response})
             else:
                 st.error("No se pudo generar una respuesta.")

     except Exception as e:
         st.error(f"⚠️ Error al generar la respuesta: {e}")

st.markdown("---")

