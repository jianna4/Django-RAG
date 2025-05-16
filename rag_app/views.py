import os
from django.shortcuts import render
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from .models import UploadedPDF
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Directory setup
UPLOAD_DIR = "pdfs/"
TEXT_DIR = "data/"
VECTORSTORE_DIR = "rag_app/vectorstores/"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_pdf(request):
    if 'file' not in request.FILES:
        return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        pdf = request.FILES['file']
        
        # Validate file extension
        if not pdf.name.lower().endswith('.pdf'):
            return Response({"error": "Only PDF files are allowed"}, status=status.HTTP_400_BAD_REQUEST)

        # Save to database
        UploadedPDF.objects.create(title=pdf.name, file=pdf)

        # Save uploaded PDF
        file_path = os.path.join(UPLOAD_DIR, pdf.name)
        with open(file_path, 'wb+') as dest:
            for chunk in pdf.chunks():
                dest.write(chunk)

        # Extract text
        try:
            reader = PdfReader(file_path)
            full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
            
            if not full_text.strip():
                return Response({"error": "PDF contains no extractable text"}, status=status.HTTP_400_BAD_REQUEST)

            # Save extracted text
            text_path = os.path.join(TEXT_DIR, f"{pdf.name}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            # Process with LangChain
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.create_documents([full_text])

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(VECTORSTORE_DIR)

            return Response({
                "message": "PDF uploaded and processed successfully",
                "text_path": text_path,
                "vectorstore": VECTORSTORE_DIR
            }, status=status.HTTP_201_CREATED)

        except PdfReadError:
            return Response({"error": "Invalid PDF file"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": f"Text processing failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def ask_question(request):
    question = request.data.get("question", "").strip()
    
    if not question:
        return Response({"error": "Question cannot be empty"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Load embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings)
        retriever = vectorstore.as_retriever()

        # Initialize LLM and QA chain
        llm = Ollama(model="tinyllama")
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Get answer
        answer = qa.run(question)
        return Response({"answer": answer})

    except FileNotFoundError:
        return Response({"error": "No processed documents found. Upload a PDF first."}, 
                      status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({"error": f"Question processing failed: {str(e)}"}, 
                      status=status.HTTP_500_INTERNAL_SERVER_ERROR)