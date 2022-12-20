
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from datetime import datetime
from django.views.decorators.csrf import csrf_exempt # 跳過Csrf保護
from django.views.decorators.csrf import csrf_protect # csrf 保護
from django.views.decorators.csrf import ensure_csrf_cookie # 瀏覽器cookie加入token
from django.views.decorators.csrf import requires_csrf_token # 这个装饰器类似csrf_protect，一样要进行csrf验证，但是它不会拒绝发送过来的请求。
import json
import numpy as np
import pandas as pd
from sklearn import datasets

# Create your views here.
# 訪問這個模板時，在cookie裡加入token
@ensure_csrf_cookie
def home(request): 
    return render(request,"index.html",) 


@requires_csrf_token
def get_dataset(request):
    resp = {}
    post_data = json.loads(request.body)
    datasets_name = post_data["datasset_name"]
    if(datasets_name == "房價資料"):
        boston_data = datasets.load_boston()
        data_df= pd.DataFrame(data=boston_data.data,columns= boston_data.feature_names)
        data_df["target"] = boston_data.target

        resp["ok"] = True
        resp["descr"] = boston_data.DESCR
        resp["rowdata"]  = data_df.to_json()
        resp["feature_names"] = pd.Series(boston_data.feature_names).append(pd.Series("TARGET")).tolist()
        resp["data"] = boston_data.data.tolist() #ndarray need to list
        resp["target"] = boston_data.target.tolist()
    return JsonResponse(resp)
  
