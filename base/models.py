from django.db import models

class Joblisting(models.Model):
    jobpost = models.TextField(max_length=255, null=True, blank=True)
    Title = models.CharField(max_length=255, null=True, blank=True)
    Company = models.CharField(max_length=255, null=True, blank=True)
    StartDate = models.CharField(max_length=255,null=True, blank=True)
    Duration = models.CharField(max_length=50, null=True, blank=True)
    Location = models.CharField(max_length=255, null=True, blank=True)
    JobDescription = models.TextField(null=True, blank=True)
    JobRequirement = models.TextField(null=True, blank=True)
    RequiredQual = models.TextField(null=True, blank=True)
    Salary = models.CharField(max_length=50, null=True, blank=True)
    AboutC = models.TextField(null=True, blank=True)
    Prediction = models.CharField(max_length= 255, null=True, blank=True)

    def __str__(self):
        return self.Title
    