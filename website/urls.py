from django.shortcuts import render
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register_user, name='register_user'),
    path('login/', views.login_user, name='login_user'),
    path('logout/', views.logout_user, name='logout_user'),
    path('delete-account/', views.delete_account, name='delete_account'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('video/<int:video_id>/edit-tags/', views.edit_video_tags, name='edit_video_tags'),
    path('delete-summary/<int:summary_id>/', views.delete_summary, name='delete_summary'),
    path('delete-video/<int:video_id>/', views.delete_video, name='delete_video'),
    path('review/<int:summary_id>/', views.add_or_edit_review, name='add_or_edit_review'),
    path('review/delete/<int:review_id>/', views.delete_review, name='delete_review'),
    path('catalog/', views.catalog, name='catalog'),
path('favorite/add/<int:video_id>/', views.add_favorite, name='add_favorite'),
path('favorite/remove/<int:video_id>/', views.remove_favorite, name='remove_favorite'),
]
