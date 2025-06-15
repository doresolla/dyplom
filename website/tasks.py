from celery import shared_task
from .mainAction import generate_summary
from .models import Audio, Summary, Format, Algo, User
import os
from django.core.mail import send_mail
from django.conf import settings

@shared_task(bind=True)
def run_summary_task(self, audio_id, algo, format, ratio):
    audio = Audio.objects.get(id=audio_id)
    video = audio.video
    algo_obj = algo
    format_obj = format
    user = video.author
    print('---------------------------------------------------')
    print(f'user = {user}')
    print(f'user.id = {user.user_id}')
    print('---------------------------------------------------')

    print(f"[TASK] Start generating summary for Video ID {video.id}")
    print(f'ALGORITHM = {algo_obj}')
    summary_path, error_message = generate_summary(
        video.video_path, video.id, format=format_obj, ratio=ratio, algo=algo_obj)

    if error_message:
        print(f"[TASK ERROR] Error generating summary for Audio ID {audio_id}: {error_message}")

        audio = Audio.objects.get(id=audio_id)
        audio.error_message = error_message
        audio.save()
        return {
            'status': 'error',
            'message': error_message
        }

    if summary_path:
        # Добавляем Summary в БД:
        if user is not None:
            print(f'user = {user}')
            print(f'user.id = {user.user_id}')

            Summary.objects.create(
                audio=audio,
                file_path=os.path.abspath(summary_path),
                format=Format.objects.get(format=format_obj),
                algorithm=Algo.objects.get(algo=algo_obj),
                user=user
            )
            summary_dir = os.path.dirname(summary_path)
            audio_basename = os.path.basename(summary_dir)
            audio.transcription_path =  os.path.join(summary_dir, f"{audio_basename}.wav")
        else:
            print(f'user is {user}')

        print(f"[TASK] Summary saved to DB for Audio ID {audio_id}")
        if user and user.email:
            try:
                send_mail(
                    subject="Конспект готов",
                    message=f"Конспект по видео '{video.title}' готов! Вы можете посмотреть его в личном кабинете.",
                    from_email=settings.EMAIL_HOST_USER,
                    recipient_list=[user.email],
                    fail_silently=False,
                )
                print(f'Отправлено сообщение на {user.email}: {video.title}')
                return {
                    'status': 'success',
                    'summary_path': os.path.abspath(summary_path)
                }
            except Exception as e:
                print(f'Ошибка во время отправки сообщения: {e}')

    # если и summary_path == None и error_message == None (маловероятно, но на всякий случай):
    print(f"[TASK ERROR] Unknown error occurred for Audio ID {audio_id}")

    return {
        'status': 'error',
        'message': 'Unknown error'
    }
