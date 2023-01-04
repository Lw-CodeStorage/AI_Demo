
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
from sklearn import preprocessing 
import base64
from io import BytesIO
import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt

import seaborn as sns

boston_data = datasets.load_boston()
data_df= pd.DataFrame(data=boston_data.data,columns= boston_data.feature_names)
data_df["TARGET"] = boston_data.target

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
       
        resp["descr"] = boston_data.DESCR

        resp["origin_column"] = pd.Series(boston_data.feature_names).append(pd.Series("TARGET")).tolist()
        resp["origin_rowdata"]  = data_df.to_dict('records')

        data_statistics = data_df.describe()
        data_statistics["STATISTICS"] = ["count","mean","std","min","25%","50%","75%","max"]
        resp["descript_column"] = data_statistics.columns.to_list()
        resp["descript_rowdata"] = data_statistics.to_dict('records')
    return JsonResponse(resp)

@requires_csrf_token
def data_distributed(request):
    data_df.hist(alpha=0.6,figsize=(6,6))
    plt.tight_layout()
    sio = BytesIO()
    plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
    data = base64.encodebytes(sio.getvalue()).decode()
    src = 'data:image/png;base64,' + str(data)
    # 记得关闭，不然画出来的图是重复的
    plt.close()
    resp={}
    resp["data"] = src
    return JsonResponse(src,safe=False)

@requires_csrf_token
def chart(request):
    #matplotlib.use('Agg')  # 不出现画图的框
    sns.set_style("darkgrid")
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 这两行用来显示汉字
    plt.rcParams['axes.unicode_minus'] = False
    # 關係分布
    #sns.regplot(x=data_df["RM"], y=data_df["TARGET"],color="red")
    # 全部分布狀況
    #data_df.hist(alpha=0.6,figsize=(6,6))
    # plt.tight_layout()
    # 長條圖 + 核密度
    # sns.displot(data_df['TARGET'],  kde=True, aspect=1.5)
   
    min_max_scaler = preprocessing.MinMaxScaler()
    column_sels = ["CRIM","ZN","INDUS",
    "CHAS","NOX","RM",
    "AGE","DIS","RAD",
    "TAX","PTRATIO","B",
    "LSTAT"]
    x = data_df.loc[:, column_sels]
    y = data_df["TARGET"]
    x = pd.DataFrame(data=min_max_scaler.fit_transform(x),columns=column_sels)
    fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(16,12))
    axs = axs.flatten()
    for i, k in enumerate(column_sels):
        sns.regplot(
            y=y,
            x=x[k],
            marker="+", 
            scatter_kws={"color":"red","alpha":0.3,"s":45},
            line_kws={"color":"#00ffff","alpha":1,"lw":4},
            ax=axs[i]
        )
    #更改 Matplotlib 子圖大小和間距
    fig.tight_layout()

    sio = BytesIO()
    plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
    data = base64.encodebytes(sio.getvalue()).decode()
    src = 'data:image/png;base64,' + str(data)
    # 记得关闭，不然画出来的图是重复的
    plt.close()
    resp={}
    resp["data"] = src
    return JsonResponse(src,safe=False)

@requires_csrf_token
def chart_pearson(request):
    sns.set(rc={"figure.figsize":(10,5)})
    sns.heatmap(data=data_df.corr(),cmap="RdBu", #cmap="Greens"
    annot_kws={"size":12},
    annot=True,
    fmt=".2f")
    plt.tight_layout()
    sio = BytesIO()
    plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
    data = base64.encodebytes(sio.getvalue()).decode()
    src = 'data:image/png;base64,' + str(data)
    # 记得关闭，不然画出来的图是重复的
    plt.close()
    resp={}
    resp["data"] = src
    return JsonResponse(src,safe=False)
