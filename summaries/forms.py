from __future__ import annotations

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import LectureNote, Rating


class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        label='Email',
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'name@example.com',
        }),
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Логин для входа',
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].label = 'Логин'
        self.fields['password1'].label = 'Пароль'
        self.fields['password2'].label = 'Повторите пароль'
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Минимум 8 символов',
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Повторите пароль',
        })

    def clean_email(self):
        email = self.cleaned_data['email'].strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError('Пользователь с таким email уже зарегистрирован.')
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user


class LectureNoteUploadForm(forms.ModelForm):
    class Meta:
        model = LectureNote
        fields = ['title', 'video_file', 'llm_model']
        widgets = {
            'title': forms.TextInput(attrs={'placeholder': 'Например: Лекция 1. Вероятность', 'class': 'form-control'}),
            'video_file': forms.ClearableFileInput(attrs={'accept': 'video/*', 'class': 'form-control'}),
            'llm_model': forms.Select(attrs={'class': 'form-control'}),
        }

    def clean_video_file(self):
        video = self.cleaned_data['video_file']
        allowed = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v']
        name = video.name.lower()
        if not any(name.endswith(ext) for ext in allowed):
            raise forms.ValidationError('Загрузите видеофайл: mp4, mkv, avi, mov, webm или m4v.')
        return video


class RatingForm(forms.ModelForm):
    class Meta:
        model = Rating
        fields = ['score', 'comment']
        widgets = {
            'score': forms.Select(choices=[(i, f'{i}') for i in range(1, 6)], attrs={'class': 'form-control'}),
            'comment': forms.Textarea(attrs={'rows': 3, 'placeholder': 'Что получилось хорошо, а что нужно улучшить?', 'class': 'form-control'}),
        }
