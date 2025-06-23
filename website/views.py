import os
from datetime import datetime
from pytubefix import YouTube
import requests
from django.conf import settings
from django.views.decorators.http import require_POST
from django.contrib.auth.hashers import check_password
from django.shortcuts import render, get_object_or_404, redirect
from django.db.models import Avg
from favorites.favorites import Favorites

from .forms import VideoTagForm, SummaryReviewForm, UserRegistrationForm, LoginForm, VideoUploadForm,SummaryAlgoFormatForm, UserSettingsForm
from .models import Summary, Video, Audio, SummaryReview, User, VideoOwnership, Tag, VideoTag, Algo
from .tasks import run_summary_task

from website.video import download, get_video_duration, extract_thumbnail, PROXIES, read_file, check_title


def home(request):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first() if user_id else None
    username = user.username if user else None
    success_message = None
    error_message = None
    video_form = VideoUploadForm()

    if request.method == 'POST':

        form = VideoUploadForm(request.POST, request.FILES)
        algo_form = SummaryAlgoFormatForm(request.POST)
        print("=== POST получен ===")
        print(f"file: {form.data.get('file')}")
        print(f"url: {form.data.get('url')}")
        print(f"title: {form.data.get('title')}")
        print(f"form.is_valid: {form.is_valid()}")
        print(f"algo_form.is_valid: {algo_form.is_valid()}")
        print(f"errors: {form.errors}")

        if form.is_valid() and algo_form.is_valid():
            print('IF')
            title = form.cleaned_data['title']
            file = form.cleaned_data.get('file')
            print(f"file: {file}")
            url = form.cleaned_data.get('url')
            video_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
            algo_obj = algo_form.cleaned_data['algo']
            format_obj = algo_form.cleaned_data['format']
            ratio = algo_form.cleaned_data.get('ratio') or 0.5
            print(f'url = d{url}d')
            os.makedirs(video_dir, exist_ok=True)
            if file and not title:
                error_message = "Укажите название для локального видео."
                return render(request, 'home.html', {
                    'form': form,
                    'algo_format_form': algo_form,
                    'user_name': username,
                    'message_to_user': error_message
                })
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
                        'user_name': username,
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
                        'user_name': username,
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
                    error_message = f"URL-{url}: Ошибка: не удалось скачать файл. {e}"
                    print(error_message)
                    return render(request, 'home.html', {
                        'form': form,
                        'user_name': username,
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
                    'user_name': username,
                    'message_to_user': error_message
                })
            else:
                video = Video.objects.create(
                    author=user if user else User.objects.filter(pk=2).first(),
                    title=title,
                    video_path=video_path,
                    url=url or '',
                    duration=duration,
                    uploaded_at=datetime.now(),
                    status=True,
                    source_name=source_name
                )
                tags = form.cleaned_data.get('tags')
                if tags:
                    for tag in tags:
                        VideoTag.objects.get_or_create(video=video, tag=tag)
                else:
                    default_tag, created = Tag.objects.get_or_create(tag_name='Неотсортировано')
                    VideoTag.objects.get_or_create(video=video, tag=default_tag)
                # Создаём сущности Audio и Summary на основе результата
                audio = Audio.objects.create(
                    video=video,
                    audio_path='',
                    transcription_path='',
                )

                run_summary_task.delay(audio.id, algo=algo_obj.algo, format = format_obj.format, ratio= ratio)

                success_message = f"Видео «{video.title}» загружено. Алгоритм: {algo_obj.algo}, формат: {format_obj.format}, ratio: {ratio}. Идёт обработка..."
                print(success_message)
                return render(request, 'home.html', {
                    'form': VideoUploadForm(),
                    'algo_format_form': algo_form,
                    'message_to_user': success_message,
                    'user_name': username,
                    'video': video
                })
    else:
        video_form = VideoUploadForm()
        initial_data = {}
        if user and hasattr(user, 'preferred_algo_id') and user.preferred_algo_id:
            preferred_algo = Algo.objects.filter(pk=user.preferred_algo_id).first()
            if preferred_algo:
                initial_data['algo'] = preferred_algo
        else:
            default_algo = Algo.objects.filter(algo='lsa').first()
            if default_algo:
                initial_data['algo'] = default_algo

        algo_form = SummaryAlgoFormatForm(initial=initial_data)

    return render(request, 'home.html', {
        'form': video_form,
        'algo_format_form': algo_form,
        'user_name': username
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
    return render(request, 'register.html', {'form': form, 'user_name': ' '})

def login_user(request):
    error = None
    user = None
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

    return render(request, 'login.html', {
        'form': form,
        'error': error,
        'user_name': user.username if user else ' ',})

def catalog(request):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first()

    query = request.GET.get('q', '')
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')

    summaries = Summary.objects.select_related('audio__video', 'format', 'algorithm').prefetch_related('reviews')
    favorites = Favorites(request)
    favorite_summaries = favorites.get_summaries()
    favorite_ids = set(favorite_summaries.values_list('id', flat=True))
    # Поиск по названию видео
    if query:
        summaries = summaries.filter(
            audio__video__title__icontains=query
        )

    # Поиск по дате (диапазон)
    if date_from and date_to:
        summaries = summaries.filter(created_at__range=[date_from, date_to])
    elif date_from:
        summaries = summaries.filter(created_at__gte=date_from)
    elif date_to:
        summaries = summaries.filter(created_at__lte=date_to)

    for summary in summaries:
        summary.video = summary.audio.video
        summary.transcript_text = summary.audio.get_transcription_text()
        summary.summary_text = summary.get_file_text()
        summary.is_favorite = summary.id in favorite_ids
        summary.my_review = SummaryReview.objects.filter(user=user, summary=summary).first()
        summary.reviews_list = summary.reviews.all()  # ← вместо summary.reviews = ...
        avg_rating = SummaryReview.objects.filter(summary=summary).aggregate(Avg('user_rating'))['user_rating__avg']
        summary.avg_rating = avg_rating if avg_rating else 0
        summary.tags = Tag.objects.filter(videotag__video=summary.audio.video)
        if summary.file_path and os.path.exists(summary.file_path):
            relative_path = os.path.relpath(summary.file_path, settings.MEDIA_ROOT)
            # на всякий случай → делаем универсально (Windows/Linux)
            relative_path = relative_path.replace(os.path.sep, '/')
            summary.download_url = settings.MEDIA_URL + relative_path
        else:
            summary.download_url = None

    return render(request, 'catalog.html', {
        'summaries': summaries,
        'user': user if user else ' ',
        'user_name': user.username if user else None,
        'query': query,
        'date_from': date_from,
        'date_to': date_to
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

    query = request.GET.get('q', '')
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')
    section = request.GET.get('section', 'videos')
    tag_query = request.GET.get('tag')

    # Все видео пользователя
    user_videos = Video.objects.filter(author=user)
    user_audios = Audio.objects.filter(video__in=user_videos)
    user_summaries = Summary.objects.filter(audio__in=user_audios).select_related('audio__video', 'format',
                                                                                                       'algorithm').prefetch_related('reviews')
    favorites = Favorites(request)
    favorite_summaries = favorites.get_summaries()
    favorite_ids = set(favorite_summaries.values_list('id', flat=True))

    if query:
        user_summaries = user_summaries.filter(audio__video__title__icontains=query)
        user_videos = user_videos.filter(title__icontains=query)
        favorite_summaries = favorite_summaries.filter(audio__video__title__icontains=query)
    if date_from and date_to:
        user_summaries = user_summaries.filter(created_at__range=[date_from, date_to])
        favorite_summaries = favorite_summaries.filter(created_at__range=[date_from, date_to])
        user_videos = user_videos.filter(uploaded_at__range=[date_from, date_to])
    elif date_from:
        user_summaries = user_summaries.filter(created_at__gte=date_from)
        favorite_summaries = favorite_summaries.filter(created_at__gte=date_from)
        user_videos = user_videos.filter(uploaded_at__gte=date_from)
    elif date_to:
        user_summaries = user_summaries.filter(created_at__lte=date_to)
        favorite_summaries = favorite_summaries.filter(created_at__lte=date_to)
        user_videos = user_videos.filter(uploaded_at__lte=date_to)
    if tag_query:
        user_summaries = user_summaries.filter(audio__video__videotag__tag__tag_name__icontains=tag_query)
        favorite_summaries = favorite_summaries.filter(audio__video__videotag__tag__tag_name__icontains=tag_query)
        user_videos = user_videos.filter(videotag__tag__tag_name__icontains=tag_query)

    # Обогащаем объекты для шаблона
    for video in user_videos:
        video.tags = Tag.objects.filter(videotag__video=video)
    for summary in user_summaries:
        summary.tags = Tag.objects.filter(videotag__video=summary.audio.video)
        summary.transcript_text = summary.audio.get_transcription_text()
        summary.summary_text = summary.get_file_text()
        summary.is_favorite = summary.id in favorite_ids
        summary.reviews_list = summary.reviews.all()
        summary.my_review = SummaryReview.objects.filter(user=user, summary=summary).first()
        avg_rating = SummaryReview.objects.filter(summary=summary).aggregate(Avg('user_rating'))['user_rating__avg']
        summary.avg_rating = avg_rating if avg_rating else 0
        # формируем download_url
        if summary.file_path:
            relative_path = os.path.relpath(summary.file_path, settings.MEDIA_ROOT)
            relative_path = relative_path.replace(os.path.sep, '/')
            summary.download_url = settings.MEDIA_URL + relative_path
        else:
            summary.download_url = None

    for summary in favorite_summaries:
        summary.tags = Tag.objects.filter(videotag__video=summary.audio.video)
        summary.transcript_text = summary.audio.get_transcription_text()
        summary.summary_text = summary.get_file_text()
        summary.reviews_list = summary.reviews.all()
        summary.is_favorite = summary.id in favorite_ids
        summary.my_review = SummaryReview.objects.filter(user=user, summary=summary).first()
        avg_rating = SummaryReview.objects.filter(summary=summary).aggregate(Avg('user_rating'))['user_rating__avg']
        summary.avg_rating = avg_rating if avg_rating else 0

        # формируем download_url
        if summary.file_path:
            relative_path = os.path.relpath(summary.file_path, settings.MEDIA_ROOT)
            relative_path = relative_path.replace(os.path.sep, '/')
            summary.download_url = settings.MEDIA_URL + relative_path
        else:
            summary.download_url = None

    return render(request, 'dashboard.html', {
        'user_videos': user_videos,
        'user_summaries': user_summaries,
        'favorite_summaries': favorite_summaries,
        'query': query,
        'section': section,
            'user_name': user.username
        })

def settings_view(request):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first()
    if not user:
        return redirect('login_user')

    # Пример — preferred_algo можно хранить в User или в отдельной модели UserSettings
    preferred_algo_id = user.preferred_algo_id if hasattr(user, 'preferred_algo_id') else None
    print(f'BEFORE preferred_algo_id={preferred_algo_id}')
    dark_theme = request.session.get('dark_theme', False)

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'save_algo':
            form = UserSettingsForm(request.POST)
            print(f'form.is_valid() ={form.is_valid()}')
            if form.is_valid():
                user.preferred_algo_id = form.cleaned_data['algo']
                print(f'user.preferred_algo_id={user.preferred_algo_id}')
                user.save()
                return redirect('settings')
            # print(request.POST)
            # preferred_algo_value = request.POST.get('algo')
            # print(f'preferred_algo_value={preferred_algo_value}')
            # if preferred_algo_value:
            #     preferred_algo_id = int(preferred_algo_value)
            #     print(f'AFTER preferred_algo_id={preferred_algo_id}')
            #     user.preferred_algo_id = preferred_algo_id
            #     user.save()

        elif action == 'save_theme':
            dark_theme = 'dark_theme' in request.POST
            request.session['dark_theme'] = dark_theme

        elif action == 'change_email':
            new_email = request.POST.get('new_email')
            user.edit_profile(email=new_email)

        elif action == 'delete_account':
            user.delete_profile()
            request.session.flush()
            return redirect('home')

        return redirect('settings')  # обновляем страницу после POST

    # Получаем список алгоритмов
    algos = Algo.objects.all()
    initial_data = {}
    if user.preferred_algo_id:
        initial_data['algo'] = user.preferred_algo_id
    form = UserSettingsForm(initial=initial_data)

    return render(request, 'settings.html', {
        'algos': algos,
        'preferred_algo_id': preferred_algo_id,
        'user_name': user.username,
        'form': form,
    })

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

    # Получаем только видео, где автор — текущий пользователь:
    video = get_object_or_404(Video, id=video_id, author=user)

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

    return render(request, 'edit_tags.html', {'form': form, 'video': video, 'user_name': user.username if user else ' '})

def add_or_edit_review(request, summary_id):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first()
    if not user:
        return redirect('login_user')
    summary = get_object_or_404(Summary, id=summary_id)
    review, created = SummaryReview.objects.get_or_create(user=user, summary=summary)

    if request.method == 'POST':
        form = SummaryReviewForm(request.POST, instance=review)
        if form.is_valid():
            form.save()
            return redirect('dashboard')
    else:
        form = SummaryReviewForm(instance=review)

    return render(request, 'review_form.html', {'form': form, 'summary': summary, 'user_name': user.username if user else ' '})

def delete_review(request, review_id):
    user_id = request.session.get('user_id')
    user = User.objects.filter(pk=user_id).first()
    if not user:
        return redirect('login_user')
    review = get_object_or_404(SummaryReview, id=review_id, author=user)
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
    return render(request, 'confirm_delete_account.html', {'user_name': user.username})

def download_summary_by_video(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    audio = getattr(video, 'audio', None)

    if not audio:
        return render(request, 'summary_not_ready.html', {'video': video, 'user_name': request.session.get('user_name')})

    summary = Summary.objects.filter(audio=audio).first()

    if not summary or not summary.file_path or not os.path.exists(summary.file_path):
        return render(request, 'summary_not_ready.html', {'video': video, 'user_name': request.session.get('user_name')})

    relative_path = os.path.relpath(summary.file_path, settings.MEDIA_ROOT).replace(os.path.sep, '/')
    download_url = settings.MEDIA_URL + relative_path

    return render(request, 'summary_ready.html', {
        'video': video,
        'download_url': download_url,
        'user_name': request.session.get('user_name')
    })