from django.db import models


# Create your models here.
class TSR(models.Model):
    filename = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_complete = models.BooleanField(default=False)
    file_path = models.TextField()
    download_url = models.TextField(default=None, blank=True, null=True)
