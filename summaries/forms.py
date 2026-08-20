from __future__ import annotations

from django import forms
from django.contrib.auth.forms import PasswordChangeForm, UserCreationForm
from django.contrib.auth.models import User

from .models import LectureNote, Review, Tag


CONTROL_CLASS = 'form-control'


class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        label='Email',
        widget=forms.EmailInput(attrs={'class': CONTROL_CLASS, 'placeholder': 'name@example.com'}),
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']
        widgets = {
            'username': forms.TextInput(attrs={'class': CONTROL_CLASS, 'placeholder': 'Логин для входа'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].label = 'Логин'
        self.fields['password1'].label = 'Пароль'
        self.fields['password2'].label = 'Повторите пароль'
        self.fields['password1'].widget.attrs.update({'class': CONTROL_CLASS, 'placeholder': 'Минимум 8 символов'})
        self.fields['password2'].widget.attrs.update({'class': CONTROL_CLASS, 'placeholder': 'Повторите пароль'})

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


class AccountEmailForm(forms.ModelForm):
    email = forms.EmailField(
        required=False,
        label='Email для уведомлений',
        widget=forms.EmailInput(attrs={'class': CONTROL_CLASS, 'placeholder': 'name@example.com'}),
        help_text='На этот адрес будут приходить уведомления о завершении обработки.',
    )

    class Meta:
        model = User
        fields = ['email']

    def __init__(self, *args, **kwargs):
        self.user = kwargs.get('instance')
        super().__init__(*args, **kwargs)

    def clean_email(self):
        email = (self.cleaned_data.get('email') or '').strip().lower()
        if email and User.objects.filter(email__iexact=email).exclude(pk=self.user.pk).exists():
            raise forms.ValidationError('Этот email уже используется другим пользователем.')
        return email


class AccountPasswordChangeForm(PasswordChangeForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        placeholders = {
            'old_password': 'Текущий пароль',
            'new_password1': 'Новый пароль',
            'new_password2': 'Повторите новый пароль',
        }
        for name, field in self.fields.items():
            field.widget.attrs.update({'class': CONTROL_CLASS, 'placeholder': placeholders[name]})


class ActiveTagMultipleChoiceField(forms.ModelMultipleChoiceField):
    def label_from_instance(self, obj: Tag) -> str:
        return obj.name


class LectureNoteUploadForm(forms.ModelForm):
    tags = ActiveTagMultipleChoiceField(
        queryset=Tag.objects.none(),
        required=False,
        label='Теги',
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'tag-checkbox-list'}),
        help_text='Доступные теги задаёт администратор.',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['video_file'].required = True
        self.fields['tags'].queryset = Tag.objects.filter(is_active=True).order_by('name')

    class Meta:
        model = LectureNote
        fields = ['title', 'video_file', 'llm_model', 'tags']
        widgets = {
            'title': forms.TextInput(attrs={'placeholder': 'Например: Лекция 1. Вероятность', 'class': CONTROL_CLASS}),
            'video_file': forms.ClearableFileInput(attrs={'accept': 'video/*', 'class': CONTROL_CLASS}),
            'llm_model': forms.Select(attrs={'class': CONTROL_CLASS}),
        }

    def clean_video_file(self):
        video = self.cleaned_data['video_file']
        allowed = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v']
        if not any(video.name.lower().endswith(ext) for ext in allowed):
            raise forms.ValidationError('Загрузите видеофайл: mp4, mkv, avi, mov, webm или m4v.')
        return video


class LectureNoteTagsForm(forms.ModelForm):
    tags = ActiveTagMultipleChoiceField(
        queryset=Tag.objects.none(),
        required=False,
        label='Теги',
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'tag-checkbox-list'}),
    )

    class Meta:
        model = LectureNote
        fields = ['tags']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        active_tags = Tag.objects.filter(is_active=True)
        if self.instance and self.instance.pk:
            active_tags = (active_tags | self.instance.tags.all()).distinct()
        self.fields['tags'].queryset = active_tags.order_by('name')


class ReviewForm(forms.ModelForm):
    class Meta:
        model = Review
        fields = ['score', 'comment']
        widgets = {
            'score': forms.Select(choices=[(i, f'{i} из 5') for i in range(1, 6)], attrs={'class': CONTROL_CLASS}),
            'comment': forms.Textarea(attrs={
                'rows': 4,
                'placeholder': 'Опишите качество текста, структуры и подобранных изображений…',
                'class': CONTROL_CLASS,
            }),
        }
