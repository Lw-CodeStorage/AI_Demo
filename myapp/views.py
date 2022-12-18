
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from datetime import datetime
from django.views.decorators.csrf import csrf_exempt # 跳過Csrf保護
from django.views.decorators.csrf import csrf_protect # csrf 保護
from django.views.decorators.csrf import ensure_csrf_cookie # 瀏覽器cookie加入token
import json
import numpy as np
import pandas as pd
from sklearn import datasets

# Create your views here.
# 訪問這個模板時，在cookie裡加入token
@ensure_csrf_cookie
def home(request): 
    return render(request,"index.html",) 


@csrf_exempt
def get_dataset(request):
    resp = {}
    post_data = json.loads(request.body)
    datasets_name = post_data["datasset_name"]
    if(datasets_name == "房價資料"):
        boston_data = datasets.load_boston()
        resp["ok"] = True
        resp["descr"] = boston_data.DESCR
        resp["feature_names"] = "boston_data.feature_names"
        resp["data"] = "boston_data.data"
        resp["target"] = [1,2,3]
    # 看起來是 sklearn 裡面的 numpuy ndarray 轉 json 發生問題   
    return JsonResponse(resp)
  
