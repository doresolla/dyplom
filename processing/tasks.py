# processing/tasks.py

from celery import shared_task
from django.core.mail import send_mail
from .models import ProcessingRequest


@shared_task
def processing_request_created(request_id):
    request_obj = ProcessingRequest.objects.get(id=request_id)

    subject = f"Запрос на обработку видео №{request_obj.id}"
    message = f"Здравствуйте, {request_obj.user.username}!\n\n" \
              f"Ваш запрос на обработку видео '{request_obj.video.title}' принят в обработку.\n" \
              f"Дата создания: {request_obj.requested_at.strftime('%d.%m.%Y %H:%M')}."

    send_mail(subject, message, 'admin@project.com', [request_obj.user.email])
