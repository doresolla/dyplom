import os
from celery import Celery

# Указываем Django настройки
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trpo.settings')

app = Celery('dyplom')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()