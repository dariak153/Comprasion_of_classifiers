from django.db import models


class Dataset(models.Model):
    SEX_CHOICES = (('male', 'male'), ('female', 'female'))
    pclass = models.SmallIntegerField()
    name = models.CharField(max_length=50)
    sex = models.CharField(max_length=6, choices=SEX_CHOICES)
    age = models.FloatField()
    sibsp = models.SmallIntegerField()
    parch = models.SmallIntegerField()
    ticket = models.CharField(max_length=18)
    fare = models.FloatField()
    cabin = models.CharField(max_length=15)
    embarked = models.CharField(max_length=1)


class PredictorsResults(models.Model):
    predictor = models.CharField(max_length=43)
    cross_val_score = models.FloatField()
    accuracy_score = models.FloatField()
    best_estimators = models.CharField(max_length=227)
    features = models.CharField(max_length=281)