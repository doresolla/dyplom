# -*- coding: utf-8 -*-
from django.test import TestCase
from website.models import User, Video, Audio, Summary, SummaryReview, Format, Algo
from website.forms import UserRegistrationForm, VideoUploadForm
from django.urls import reverse
from website.text import Text
import os

# --- тест моделей ---
class UserModelTest(TestCase):
    def test_create_user(self):
        user = User.objects.create(
            email='test@example.com',
            username='testuser',
            phone_number='123456789',
            password='hashed_password'
        )
        self.assertEqual(user.email, 'test@example.com')

class VideoModelTest(TestCase):
    def setUp(self):
        self.user = User.objects.create(
            email='test2@example.com', username='testuser2', phone_number='987654321', password='hashed')
    def test_create_video(self):
        video = Video.objects.create(
            author=self.user,
            title='Test Video',
            video_path='/path/to/video.mp4',
            duration=300
        )
        self.assertEqual(video.title, 'Test Video')

# --- тест форм ---
class UserRegistrationFormTest(TestCase):
    def test_valid_data(self):
        form_data = {
            'email': 'user1@example.com',
            'username': 'user1',
            'phone_number': '12345',
            'password': 'pass12345',
            'confirm_password': 'pass12345'
        }
        form = UserRegistrationForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_password_mismatch(self):
        form_data = {
            'email': 'user2@example.com',
            'username': 'user2',
            'phone_number': '12345',
            'password': 'pass123',
            'confirm_password': 'otherpass'
        }
        form = UserRegistrationForm(data=form_data)
        self.assertFalse(form.is_valid())

class VideoUploadFormTest(TestCase):
    def test_validation_empty(self):
        form = VideoUploadForm(data={})
        self.assertFalse(form.is_valid())

# --- тест представлений ---
class RegisterUserViewTest(TestCase):
    def test_register_user_get(self):
        response = self.client.get(reverse('register_user'))
        self.assertEqual(response.status_code, 200)

# --- тест обработки текста ---
class TextProcessingTest(TestCase):
    def test_tokenize(self):
        test_file = 'tests/test_data/sample.txt'
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Это пример текста. Здесь два предложения.")
        text_obj = Text(abs_name=test_file, video_id=1)
        text_obj.tokenize()
        self.assertGreater(len(text_obj.sentences), 0)
