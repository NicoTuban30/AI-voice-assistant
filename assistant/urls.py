# assistant/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),  # Homepage with UI
    path("voice-command/", views.voice_command_view, name="voice_command"),
]
