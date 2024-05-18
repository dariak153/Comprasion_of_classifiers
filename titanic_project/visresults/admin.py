from django.contrib import admin
from .models import Dataset, PredictorsResults


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked')
    list_filter = ('pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked')
    search_fields = ('pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked')


@admin.register(PredictorsResults)
class PredictorsResultsAdmin(admin.ModelAdmin):
    list_display = ('predictor', 'features', 'cross_val_score', 'accuracy_score', 'best_estimators')
    list_filter = ('predictor', 'cross_val_score', 'accuracy_score')
    search_fields = ('predictor', 'cross_val_score', 'accuracy_score', 'best_estimators')
    ordering = ('accuracy_score', 'predictor')

