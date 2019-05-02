from django.db import models

# Create your models here.

class Team(models.Model):
    name = models.CharField(max_length = 200)

    def __str__(self):
        return self.name


'''
class param(models.Model):
    weights = models.DecimalField(default = 0)
    featureMean = models.DecimalField(default = 0)
'''
