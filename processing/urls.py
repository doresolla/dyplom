from django.urls import path
from . import views

app_name = 'processing'

urlpatterns = [
    path('create/<int:video_id>/', views.create_processing_request, name='create'),
]