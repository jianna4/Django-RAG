from django.db import models
from django.db import models
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class UploadedPDF( models.Model ):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='pdfs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)  # Original save
        
        # Process only new uploads (not on every save)
        if not hasattr(self, '_processing_done'):
            self._process_pdf()
            self._processing_done = True
    
    def _process_pdf(self):
        # 1. Ensure directories exist
        os.makedirs("data/", exist_ok=True)
        os.makedirs("rag_app/vectorstores/", exist_ok=True)
        
        # 2. Extract text
        pdf_path = self.file.path
        reader = PdfReader(pdf_path)
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        
        # 3. Save text
        text_path = f"data/{os.path.basename(pdf_path)}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        
        # 4. Create embeddings
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([full_text])
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("rag_app/vectorstores/")
