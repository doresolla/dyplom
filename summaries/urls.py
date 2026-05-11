from django.urls import path

from . import views

urlpatterns = [
    path('accounts/register/', views.register, name='register'),
    path('', views.public_notes, name='public_notes'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('upload/', views.upload_note, name='upload_note'),
    path('notes/<int:pk>/', views.note_detail, name='note_detail'),
    path('notes/<int:pk>/logs/', views.processing_logs, name='processing_logs'),
    path('notifications/', views.notifications_list, name='notifications'),
]
