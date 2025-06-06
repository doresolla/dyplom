import os
from datetime import datetime
from pytubefix import YouTube
import requests
from django.conf import settings
from django.views.decorators.http import require_POST
from django.contrib.auth.hashers import check_password
from django.shortcuts import render, get_object_or_404, redirect

from .mainAction import generate_summary

from .forms import VideoTagForm, SummaryReviewForm, UserRegistrationForm, LoginForm, VideoUploadForm
from .models import Summary, Video, Audio, SummaryReview, User, VideoOwnership

from website.video import download, get_video_duration, extract_thumbnail, PROXIES, read_file, check_title

def home(request):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first()
    success_message = None
    error_message = None
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid() and user:
            title = form.cleaned_data['title']
            file = form.cleaned_data.get('file')
            url = form.cleaned_data.get('url')
            video_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
            os.makedirs(video_dir, exist_ok=True)

            if file:
                # Загрузка локального файла
                filename = file.name
                video_path = os.path.join(video_dir, filename)
                thumbnail_path = os.path.join(video_dir, filename.rsplit('.', 1)[0] + '_thumb.jpg')
                with open(video_path, 'wb+') as dest:
                    for chunk in file.chunks():
                        dest.write(chunk)
                    print(f"Скачано видео {video_path}")
                duration, error_message = get_video_duration(video_path)
                if error_message:
                    return render(request, 'home.html', {
                        'form': VideoUploadForm(),  # создать новую форму
                        'user_name': user.username,
                        'message_to_user': error_message
                    })
                source_name = "Локальная загрузка"
                try:
                    extract_thumbnail(video_path, thumbnail_path)
                except Exception as e:
                    print(e)
                print('Обработка завершена')
            elif url:
                video_object = Video.objects.filter(url=url).first()
                if video_object:
                    error_message = f"Видео '{video_object.title}' уже есть в БД"
                    print(error_message)
                    return render(request, 'home.html', {
                        'form': form,
                        'user_name': user.username if user else None,
                        'message_to_user': error_message
                    })
                print('Скачивание видео начато')
                try:
                    audio_path, duration, _, message_to_user = download(url)
                    filename = check_title(audio_path.rsplit('.', 1)[0]) + '.mp4'
                    print(audio_path, duration, filename)
                    video_path = os.path.join(video_dir, filename)
                    thumbnail_path = os.path.join(video_dir, filename.rsplit('.', 1)[0] + '_thumb.jpg')
                    yt = YouTube(url, proxies=PROXIES)
                    print(f'yt.thumbnail_url={yt.thumbnail_url}')
                    thumb_url = yt.thumbnail_url
                    r = requests.get(thumb_url)
                    if r.status_code == 200:
                        with open(thumbnail_path, 'wb') as f:
                            f.write(r.content)
                        print(f"Сохранено превью: {thumbnail_path}")
                    title = yt.title
                    source_name = yt.author  # название канала
                except Exception as e:
                    error_message = f"NAME-{filename} - Ошибка: не удалось скачать файл. {e}"
                    print(error_message)
                    return render(request, 'home.html', {
                        'form': form,
                        'user_name': user.username if user else None,
                        'message_to_user': error_message
                    })

            print(f'Видео: {video_path} длительностью {duration} секунд')
            if duration < 180 or duration > 10800:
                error_message = f"Ошибка: длительность видео {duration} сек. должна быть от 3 минут до 3 часов."
                print(error_message)

                os.remove(video_path)
                print(f'Удалено {video_path}')

                return render(request, 'home.html', {
                    'form': form,
                    'user_name': user.username if user else None,
                    'message_to_user': error_message
                })
            else:
                video = Video.objects.create(
                    author=user,
                    title=title,
                    video_path=video_path,
                    url=url or '',
                    duration=duration,
                    uploaded_at=datetime.now(),
                    status=True,
                    source_name=source_name
                )
                # Генерация конспекта через существующую функцию
                summary_path, error_message = generate_summary(url=video_path, video_id=str(video.id), format='docx', ratio=0.5)
                if error_message:
                    return render(request, 'home.html', {
                        'form': VideoUploadForm(),  # создать новую форму
                        'user_name': user.username,
                        'message_to_user': error_message
                    })
                else:
                # Создаём сущности Audio и Summary на основе результата
                    audio_path = video_path.replace('.mp4', '.wav')
                    transcript_path = video_path.replace('.mp4', '.txt')  # результат распознавания

                    audio = Audio.objects.create(
                        video=video,
                        audio_path=audio_path,
                        transcription_path=transcript_path,
                    )

                    Summary.objects.create(
                        audio=audio,
                        file_path=summary_path,
                        format='docx'
                    )

                    success_message = f"Видео «{title}» обработано успешно. Длительность: {duration} сек."
                    print(success_message)
                    return render(request, 'home.html', {
                        'form': VideoUploadForm(),
                        'message_to_user': success_message,
                        'user_name': user.username
                    })
    else:
        form = VideoUploadForm()

    return render(request, 'home.html', {
        'form': form,
        'user_name': user.username if user else None,
        'message_to_user': success_message if success_message else error_message
    })
def register_user(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])  # Хешируем
            user.save()

            # Автоматический вход: вручную создаём сессию
            request.session['user_id'] = user.pk
            request.session['user_name'] = user.username
            return redirect('home')  # Перенаправление на главную
    else:
        form = UserRegistrationForm()
    return render(request, 'register.html', {'form': form})

def login_user(request):
    error = None

    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            try:
                user = User.objects.get(email=form.cleaned_data['email'])
                if check_password(form.cleaned_data['password'], user.password):
                    request.session['user_id'] = user.pk
                    request.session['user_name'] = user.username
                    return redirect('home')
                else:
                    error = "Неверный пароль"
            except User.DoesNotExist:
                error = "Пользователь не найден"
    else:
        form = LoginForm()

    return render(request, 'login.html', {'form': form, 'error': error})

def catalog(request):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first()

    summaries = Summary.objects.select_related('audio__video').prefetch_related('reviews')

    for summary in summaries:
        summary.video = summary.audio.video
        summary.transcript_text = summary.audio.get_transcription_text()
        summary.summary_text = summary.get_file_text()
        if user:
            summary.is_favorite = VideoOwnership.objects.filter(user=user, video=summary.video).exists()
            summary.my_review = SummaryReview.objects.filter(user=user, summary=summary).first()

    return render(request, 'catalog.html', {
        'summaries': summaries,
        'user': user
    })


@require_POST
def add_favorite(request, video_id):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first()
    if user:
        video = get_object_or_404(Video, pk=video_id)
        VideoOwnership.objects.get_or_create(user=user, video=video)
    return redirect('catalog')

@require_POST
def remove_favorite(request, video_id):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first()
    if user:
        VideoOwnership.objects.filter(user=user, video_id=video_id).delete()
    return redirect('catalog')

def dashboard(request):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first()
    if not user:
        return redirect('login_user')
    # Все видео, загруженные пользователем
    user_videos = Video.objects.filter(author=user)

    # Все аудиофайлы, связанные с этими видео
    user_audios = Audio.objects.filter(video__in=user_videos)

    # Все конспекты, полученные с этих аудио
    user_summaries = Summary.objects.filter(audio__in=user_audios).select_related('audio__video')

    # Чтение текстов из файлов
    for summary in user_summaries:
        summary.transcript_text = read_file(summary.audio.transcription_path)
        summary.summary_text = read_file(summary.file_path)

    context = {
        'user_videos': user_videos,
        'user_summaries': user_summaries,
    }

    return render(request, 'dashboard.html', context)
@require_POST
def delete_summary(request, summary_id):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first()
    if not user:
        return redirect('login_user')
    summary = get_object_or_404(Summary, id=summary_id)
    summary.delete()
    return redirect('dashboard')
@require_POST
def delete_video(request, video_id):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first()
    if not user:
        return redirect('login_user')
    video = get_object_or_404(Video, id=video_id)
    video.delete()
    return redirect('dashboard')

def edit_video_tags(request, video_id):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first()
    if not user:
        return redirect('login_user')
    video = get_object_or_404(Video, id=video_id, author=user)

    if request.method == 'POST':
        form = VideoTagForm(request.POST, video=video)
        if form.is_valid():
            form.save()
            return redirect('dashboard')
    else:
        form = VideoTagForm(video=video)

    return render(request, 'edit_tags.html', {'form': form, 'video': video})

def add_or_edit_review(request, summary_id):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first()
    if not user:
        return redirect('login_user')
    summary = get_object_or_404(Summary, id=summary_id)
    review, created = SummaryReview.objects.get_or_create(author=user, summary=summary)

    if request.method == 'POST':
        form = SummaryReviewForm(request.POST, instance=review)
        if form.is_valid():
            form.save()
            return redirect('dashboard')
    else:
        form = SummaryReviewForm(instance=review)

    return render(request, 'review_form.html', {'form': form, 'summary': summary})

def delete_review(request, review_id):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first()
    if not user:
        return redirect('login_user')
    review = get_object_or_404(SummaryReview, id=review_id, author=user)
    summary_id = review.summary.id
    review.delete()
    return redirect('dashboard')

def logout_user(request):

    request.session.flush()  # удаляет все данные сессии
    return redirect('home')

def delete_account(request):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first()
    if request.method == 'POST':
        user.delete()
        request.session.flush()  # Очистка сессии
        return redirect('home')
    return render(request, 'confirm_delete_account.html', {'user': user})
