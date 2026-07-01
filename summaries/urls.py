from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('accounts/register/', views.register, name='register'),
    path('accounts/profile/', views.account_profile, name='account_profile'),
    path('accounts/password-change/', views.AccountPasswordChangeView.as_view(), name='password_change'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('favorites/', views.favorites, name='favorites'),
    path('upload/', views.upload_note, name='upload_note'),
    path('notes/<int:pk>/', views.note_detail, name='note_detail'),
    path('notes/<int:pk>/tags/', views.update_note_tags, name='update_note_tags'),
    path('notes/<int:pk>/favorite/', views.toggle_favorite, name='toggle_favorite'),
    path('notes/<int:pk>/review/', views.save_review, name='save_review'),
    path('notes/<int:pk>/review/delete/', views.delete_review, name='delete_review'),
    path('notes/<int:pk>/delete/', views.delete_note, name='delete_note'),
    path('notes/<int:pk>/files/<str:file_kind>/', views.note_file, name='note_file'),
    path('notes/<int:pk>/logs/', views.processing_logs, name='processing_logs'),
    path('notifications/', views.notifications_list, name='notifications'),
]
