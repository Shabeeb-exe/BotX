from django.db import models
from django.core.validators import URLValidator

class ExtractedURL(models.Model):
    url = models.URLField(max_length=500, validators=[URLValidator()])
    base_url = models.URLField(max_length=500)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['base_url', 'created_at']),
        ]
        ordering = ['-created_at']

    def __str__(self):
        return self.url