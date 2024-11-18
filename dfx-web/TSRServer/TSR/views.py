import json

from django.http import JsonResponse, FileResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import time, math, hashlib
from .models import TSR
import zipfile
import os

# Create your views here.


def index(request):
    return render(request, 'index.html')


def download(request):
    code = request.GET.get("code", None)
    return render(request, 'download.html', {'code': code})


def downloader(request):
    code = request.GET.get("code", None)
    print(code)
    tsr = TSR.objects.get(download_url=code)
    file_name = tsr.filename

    file_path = tsr.file_path.split("/")
    file_path.pop()
    file_path = "/".join(file_path)
    print(file_name, file_path)

    fs = FileSystemStorage(f"{file_path}")
    response = FileResponse(fs.open(f"{file_name}.zip", "rb"), content_type="application/zip")
    response['Content-Disposition'] = f'attachment; filename="{file_name}.zip"'
    return response

@csrf_exempt
def upload(request):
    if request.method == 'POST':
        file = request.FILES['uploadFile']
        fs = FileSystemStorage()
        filename = fs.save(f"./bgProcess/files/{str(math.trunc(time.time()))}/{file.name}", file)
        print(file.name, fs.url(filename))
        tsr = TSR(filename=file.name, file_path=fs.path(filename), download_url=str(hashlib.sha256(file.name.encode()).hexdigest())+str(time.time()))
        tsr.save()
        return JsonResponse({"code": 200, "url": tsr.download_url})


def complete_check(request):
    if request.method == "GET":
        print(request.GET)
        code = request.GET.get("url", None)
        print(code)
        tsr = TSR.objects.get(download_url=code)
        if tsr is None:
            return JsonResponse({"code": 202})
        if tsr.is_complete:
            return JsonResponse({"code": 200})
        else:
            return JsonResponse({"code": 201})
