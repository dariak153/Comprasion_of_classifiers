from django.http import JsonResponse
from django.shortcuts import render
from django.shortcuts import render, get_object_or_404
from .models import Dataset, PredictorsResults
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import json
import pandas as pd

# Create your views here.



def home_page(request):
    return render(request, "visresults/base.html")


def data_explainer(request):
    object_list = Dataset.objects.all()
    if len(object_list) == 0:
        return render(request, "visresults/dataset.html", None)

    columns = list(object_list.values()[0].keys())
    data_rows = object_list.values_list()

    paginator = Paginator(data_rows, 25)
    page = request.GET.get('page')
    try:
        rows = paginator.get_page(page)
    except PageNotAnInteger:
        rows = paginator.page(1)
    except EmptyPage:
        rows = paginator.page(paginator.num_pages)
    return render(request, "visresults/dataset.html", {'page': page, 'columns': columns, 'rows': rows})


def response_generator(queryset_data, first: bool = False):
    data = {
        "labels": [f"{x.id}, {x.predictor}" for x in queryset_data],
        "values": [x.accuracy_score for x in queryset_data],
        "colors": [f"rgba(255, {(32 + i) % 254}, {(54 + i) % 220}, 0.8)" for i in range(len(queryset_data))]
    }
    if first:
        return json.dumps(data)
    return data

def results_chart(request):
    objects_lists = PredictorsResults.objects.all()
    if request.method == 'POST':
        input_data = request.POST.get('input_data')

        if input_data is None:
            data = response_generator(objects_lists.order_by('-accuracy_score'))
            return JsonResponse(data)
        elif input_data.isdigit():
            if PredictorsResults.objects.first().id < int(input_data) < PredictorsResults.objects.last().id:
                row = PredictorsResults.objects.get(id=int(input_data.strip()))
                return JsonResponse({"id": f"{row.id}", "cross_val_score": f"{row.cross_val_score}", "best_estimators": f"{row.best_estimators}", "features": f"{row.features}"})
            return JsonResponse({})
        else:
            filtered_data = PredictorsResults.objects.filter(predictor__regex=f'.*{input_data.strip()}.*').order_by('-accuracy_score')
            data = response_generator(filtered_data)
            return JsonResponse(data)

    if len(objects_lists) == 0:
        return render(request, 'visresults/predictors.html', {'chart_data': None})

    sorted_obj = objects_lists.order_by('-accuracy_score')
    data = response_generator(sorted_obj, True)

    return render(request, 'visresults/predictors.html', {'chart_data': data})


def load_titanic_dataset_to_db(filepath: str = 'visresults/data/titanic_dataset.csv'):

    if len(Dataset.objects.all()) == 0:
        df = pd.read_csv(filepath)
        df = df.fillna(-1)
        for i in range(len(df)):
            v = list(df.values[i])
            print(v)
            row = Dataset(pclass=v[0],
                          name=v[1],
                          sex=v[2],
                          age=v[3],
                          sibsp=v[4],
                          parch=v[5],
                          ticket=v[6],
                          fare=v[7],
                          cabin=v[8],
                          embarked=v[9])
            row.save()
        print(f"Saved rows: {i+1}")
    else:
        print("Titanic dataset already loaded!")


def load_results_dataset_to_db(filepaths: list[str]):

    if len(PredictorsResults.objects.all()) == 0:
        df = pd.DataFrame()
        for filepath in filepaths:
            df = pd.concat([df, pd.read_csv(filepath)])
        df = df.fillna('-1')
        for i in range(len(df)):
            v = list(df.values[i])
            print(v)
            row = PredictorsResults(predictor=v[0],
                          cross_val_score=v[1],
                          accuracy_score=v[2],
                          best_estimators=v[3],
                          features=v[4],
                         )
            row.save()
        print(f"Saved rows: {i+1}")
    else:
        print("Results dataset already loaded!")

results_files = ['visresults/data/cln_results.csv',
                 'visresults/data/cln_results_1.csv',
                 'visresults/data/cln_results_2.csv',
                 'visresults/data/cln_results_3.csv'
                 ]
load_results_dataset_to_db(results_files)
load_titanic_dataset_to_db()

# basic scaled tuned random
