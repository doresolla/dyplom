from django import forms
from .models import SummaryReview, Video, Summary, Tag, VideoTag, User, Algo, Format



class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ['email', 'username', 'phone_number', 'password']

    def clean_email(self):
        email = self.cleaned_data['email']
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("Email уже зарегистрирован")
        return email

    def clean_username(self):
        username = self.cleaned_data['username']
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("Имя пользователя уже занято")
        return username

    def clean(self):
        cleaned_data = super().clean()
        if cleaned_data.get('password') != cleaned_data.get('confirm_password'):
            raise forms.ValidationError("Пароли не совпадают")
        return cleaned_data


class LoginForm(forms.Form):
    email = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)

class SummaryAlgoFormatForm(forms.Form):
    algo = forms.ModelChoiceField(
        queryset=Algo.objects.all(),
        label='Алгоритм суммаризации'
    )
    format = forms.ModelChoiceField(
        queryset=Format.objects.all(),
        label='Формат конспекта'
    )
    ratio = forms.FloatField(
        label='Процент предложений (ratio)',
        min_value=0.1,
        max_value=1.0,
        required=False,
        initial=0.5
    )

class VideoUploadForm(forms.Form):
    title = forms.CharField(max_length=255, required=False)
    file = forms.FileField(required=False)
    url = forms.URLField(label='Ссылка на видео', required=False)

    def clean(self):
        cleaned_data = super().clean()
        file = cleaned_data.get('file')
        url = cleaned_data.get('url')

        if not file and not url:
            raise forms.ValidationError("Укажите либо файл, либо ссылку.")
        if file and url:
            raise forms.ValidationError("Можно указать только один источник: файл или ссылку.")
        return cleaned_data

    def clean_duration(self):
        duration = self.cleaned_data.get('duration')
        if duration is not None:
            if duration < 180 or duration > 10800:
                raise forms.ValidationError("Длительность видео должна быть от 3 минут до 3 часов.")
        return duration

class SummaryReviewForm(forms.ModelForm):
    user_rating = forms.IntegerField(
        required=True,
        label="Оценка (1–5)",
        min_value=1,
        max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Оценка от 1 до 5'})
    )

    text = forms.CharField(
        required=False,
        label="Комментарий",
        widget=forms.Textarea(attrs={'rows': 3, 'class': 'form-control', 'placeholder': 'Оставьте отзыв (необязательно)'})
    )

    class Meta:
        model = SummaryReview
        fields = ['user_rating', 'text']
class SummaryForm(forms.ModelForm):
    class Meta:
        model = Summary
        fields = ['format', 'algorithm']

class TagForm(forms.ModelForm):
    class Meta:
        model = Tag
        fields = ['tag_name']

class VideoTagForm(forms.Form):
    video = forms.ModelChoiceField(queryset=Video.objects.all(), widget=forms.HiddenInput())
    tags = forms.ModelMultipleChoiceField(
        queryset=Tag.objects.all(),
        widget=forms.CheckboxSelectMultiple,
        required=False
    )

    def __init__(self, *args, **kwargs):
        video = kwargs.pop('video', None)
        super().__init__(*args, **kwargs)
        if video:
            self.fields['video'].initial = video
            self.fields['tags'].initial = Tag.objects.filter(videotag__video=video)

    def save(self):
        video = self.cleaned_data['video']
        selected_tags = self.cleaned_data['tags']

        # Удаляем старые связи
        VideoTag.objects.filter(video=video).exclude(tag__in=selected_tags).delete()

        # Добавляем новые связи
        for tag in selected_tags:
            VideoTag.objects.get_or_create(video=video, tag=tag)

