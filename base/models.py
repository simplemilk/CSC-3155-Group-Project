from django.db import models

class JobPost(models.Model):
    title = models.CharField(max_length=255)
    company = models.CharField(max_length=255)
    start_date = models.DateField(null=True, blank=True)
    duration = models.CharField(max_length=50, null=True, blank=True)
    location = models.CharField(max_length=255)
    description = models.TextField()
    requirements = models.TextField()
    qualifications = models.TextField()
    salary = models.CharField(max_length=50, null=True, blank=True)
    about_company = models.TextField()
    offers = models.TextField()

    def __str__(self):
        return self.title
