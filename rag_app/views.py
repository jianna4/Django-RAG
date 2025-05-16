from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from .models import UploadedPDF
from PyPDF2 import PdfReader
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader

UPLOAD_DIR = "pdfs/"
TEXT_DIR = "data/"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_pdf(request):
    pdf = request.FILES['file']
    instance = UploadedPDF.objects.create(title=pdf.name, file=pdf)

    # Save uploaded PDF
    file_path = os.path.join(UPLOAD_DIR, pdf.name)
    with open(file_path, 'wb+') as dest:
        for chunk in pdf.chunks():
            dest.write(chunk)

    # Extract text
    reader = PdfReader(file_path)
    full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
    text_path = os.path.join(TEXT_DIR, f"{pdf.name}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    # Load, split, and embed using LangChain
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([full_text])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("rag_app/vectorstore")

    return Response({"message": "PDF uploaded and embedded successfully."})

from langchain.llms import Ollama
from langchain.chains import RetrievalQA

@api_view(['POST'])
def ask_question(request):
    question = request.data.get("question", "")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("rag_app/vectorstore", embeddings)
    retriever = vectorstore.as_retriever()

    llm = Ollama(model="llama3")  # Make sure this model is pulled
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    answer = qa.run(question)
    return Response({"answer": answer})
