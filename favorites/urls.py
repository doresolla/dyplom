from django.urls import path
from . import views

app_name = 'favorites'

urlpatterns = [
    path('', views.favorites_detail, name='favorites_detail'),
    path('add/<int:summary_id>/', views.favorites_add, name='favorites_add'),
    path('remove/<int:summary_id>/', views.favorites_remove, name='favorites_remove'),
]
