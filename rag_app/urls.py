from django.urls import path
from .views import upload_pdf , ask_question  # Import the ask_question view
urlpatterns = [
    path('upload/', upload_pdf),
    path('ask/', ask_question),  # Add this line to include the ask_question view
]
