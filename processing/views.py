# -*- coding: utf-8 -*-
from django.shortcuts import render, redirect, get_object_or_404
from .forms import ProcessingRequestForm
from .models import ProcessingRequest
from website.models import Video, Audio, Summary, User
from website.forms import SummaryAlgoFormatForm, VideoUploadForm


def create_processing_request(request, video_id):
    user = User.objects.get(pk=request.session.get('user_id'))
    if not user:
        return redirect('login_user')  # если пользователь не авторизован

    video = get_object_or_404(Video, pk=video_id)
    audio = Audio.objects.filter(video=video).first()

    algo_form = SummaryAlgoFormatForm()
    format_form = VideoUploadForm()

    if request.method == 'POST':
        algo_form = SummaryAlgoFormatForm(request.POST)
        if algo_form.is_valid():
            processing_request = ProcessingRequest.objects.create(
                user=user,
                video=video,
                audio=audio,
                algorithm=algo_form.cleaned_data['algo'],
                format=algo_form.cleaned_data['format'],
                ratio=algo_form.cleaned_data['ratio']
            )
            # после оформления можно запустить обработку
            # run_summary_task.delay(audio.id, algo=..., format=..., ratio=...)
            return render(request, 'success.html',
                          {'request_obj': processing_request, 'user_name': user.username})
    else:
        form = ProcessingRequestForm()

    return render(request, 'create.html', {
        'video': video,
        'form': algo_form,
        'user_name': user.username,
    })
