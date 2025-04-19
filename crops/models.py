from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    location = models.CharField(max_length=255)

# class CropInfo(models.Model):
#     crop_name = models.CharField(max_length=100)
#     season = models.CharField(max_length=50)
#     soil_type = models.CharField(max_length=50)
#     temperature = models.FloatField()
#     rainfall = models.FloatField()

# class CropYield(models.Model):
#     crop = models.ForeignKey(CropInfo, on_delete=models.CASCADE)
#     season = models.CharField(max_length=50)
#     yield_percentage = models.FloatField()
#     location = models.CharField(max_length=255)
