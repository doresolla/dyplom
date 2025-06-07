from celery import shared_task
from .mainAction import generate_summary
from .models import Audio, Summary, Format, Algo, User
import os
from django.core.mail import send_mail
@shared_task(bind=True)
def run_summary_task(self, audio_id, algo, format, ratio):
    audio = Audio.objects.get(id=audio_id)
    video = audio.video
    algo_obj = algo
    format_obj = format
    user = User.objects.filter(username=video.author).first()
    print(f"[TASK] Start generating summary for Video ID {video.id}")
    print(f'ALGORITHM = {algo_obj}')
    summary_path, error_message = generate_summary(
        video.video_path, video.id, format=format_obj, ratio=ratio, algo=algo_obj)

    if error_message:
        print(f"[TASK ERROR] Error generating summary for Audio ID {audio_id}: {error_message}")

        # Можно при желании записать ошибку в БД — например, в поле Audio или SummaryReview

        return {
            'status': 'error',
            'message': error_message
        }

    if summary_path:
        # Добавляем Summary в БД:
        Summary.objects.create(
            audio=audio,
            file_path=os.path.abspath(summary_path),
            format=Format.objects.get(format=format_obj),
            algorithm=Algo.objects.get(algo=algo_obj),
        )

        print(f"[TASK] Summary saved to DB for Audio ID {audio_id}")
        if user and user.email:
            send_mail(
                subject="Конспект готов",
                message=f"Конспект по видео '{video.title}' готов! Вы можете посмотреть его в личном кабинете.",
                from_email=None,
                recipient_list=[user.email],
                fail_silently=True,
            )
        return {
            'status': 'success',
            'summary_path': os.path.abspath(summary_path)
        }

    # если и summary_path == None и error_message == None (маловероятно, но на всякий случай):
    print(f"[TASK ERROR] Unknown error occurred for Audio ID {audio_id}")

    return {
        'status': 'error',
        'message': 'Unknown error'
    }