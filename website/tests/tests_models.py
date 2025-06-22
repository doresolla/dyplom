# tests/test_models.py

from django.test import TestCase
from website.models import User, Video, Audio, Summary, SummaryReview, Format, Algo

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
