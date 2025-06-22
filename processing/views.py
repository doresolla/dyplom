# -*- coding: utf-8 -*-
from django.shortcuts import render, redirect, get_object_or_404
from .forms import ProcessingRequestForm
from .models import ProcessingRequest
from website.models import Video, Audio, Summary, User


def create_processing_request(request, video_id):
    user = User.objects.get(pk=request.session.get('user_id'))
    video = get_object_or_404(Video, pk=video_id)
    audio = Audio.objects.filter(video=video).first()

    if request.method == 'POST':
        form = ProcessingRequestForm(request.POST)
        if form.is_valid():
            processing_request = form.save(commit=False)
            processing_request.user = user
            processing_request.video = video
            processing_request.audio = audio
            processing_request.save()

            # Здесь можно запустить обработку
            # run_summary_task.delay(audio.id, ...)

            return render(request, 'success.html', {'request_obj': processing_request})
    else:
        form = ProcessingRequestForm()

    return render(request, 'create.html', {
        'form': form,
        'video': video
    })
