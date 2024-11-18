from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name="index"),
    path('download', views.download, name="index"),
    path('downloader', views.downloader, name="downloader"),
    path('upload', views.upload, name="upload"),
    path('check', views.complete_check, name="complete_check"),
]
