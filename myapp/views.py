
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from datetime import datetime
from django.views.decorators.csrf import csrf_exempt # 跳過Csrf保護
from django.views.decorators.csrf import csrf_protect # csrf 保護
from django.views.decorators.csrf import ensure_csrf_cookie # 瀏覽器cookie加入token
from django.views.decorators.csrf import requires_csrf_token # 这个装饰器类似csrf_protect，一样要进行csrf验证，但是它不会拒绝发送过来的请求。
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor # neural network
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

boston_data = datasets.load_boston()
diabetes_data =  datasets.load_diabetes()


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
        # 組裝資料
        data_df= pd.DataFrame(data=boston_data.data,columns= boston_data.feature_names)
        data_df["TARGET"] = boston_data.target

        # 資料描述
        resp["descr"] = boston_data.DESCR

        # 原始資料
        resp["origin_column"] = pd.Series(boston_data.feature_names).append(pd.Series("TARGET")).tolist()
        resp["origin_rowdata"]  = data_df.to_dict('records')

        # 敘述統計資料
        data_statistics = data_df.describe()
        data_statistics["STATISTICS"] = ["count","mean","std","min","25%","50%","75%","max"]
        resp["descript_column"] = data_statistics.columns.to_list()
        resp["descript_rowdata"] = data_statistics.to_dict('records')

        # 資料分布圖
        data_distributed(resp,data_df)
        # 資料趨勢圖
        trend_chart(resp,data_df)
        # 皮爾森熱力圖
        chart_pearson(resp,data_df)
        return JsonResponse(resp)
    elif(datasets_name == "糖尿病資料"):
         # 組裝資料
        data_df= pd.DataFrame(data=diabetes_data.data,columns= diabetes_data.feature_names)
        data_df["TARGET"] = diabetes_data.target

        # 資料描述
        resp["descr"] = diabetes_data.DESCR

        # 原始資料
        resp["origin_column"] = pd.Series(diabetes_data.feature_names).append(pd.Series("TARGET")).tolist()
        resp["origin_rowdata"]  = data_df.to_dict('records')

        # 敘述統計資料
        data_statistics = data_df.describe()
        data_statistics["STATISTICS"] = ["count","mean","std","min","25%","50%","75%","max"]
        resp["descript_column"] = data_statistics.columns.to_list()
        resp["descript_rowdata"] = data_statistics.to_dict('records')

        # 資料分布圖
        data_distributed(resp,data_df)
        # 資料趨勢圖
        trend_chart(resp,data_df)
        # 皮爾森熱力圖
        chart_pearson(resp,data_df)
        return JsonResponse(resp)
   


def data_distributed(resp,data_df):
    data_df.hist(alpha=0.6,figsize=(20,5),layout=(2,7))
    plt.tight_layout()
    sio = BytesIO()
    plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.3)
    data = base64.encodebytes(sio.getvalue()).decode()
    src = 'data:image/png;base64,' + str(data)
    # 记得关闭，不然画出来的图是重复的
    plt.close()
  
    resp["distributed_image"] = src

def trend_chart(resp,data_df):
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

    x = data_df.loc[:, data_df.columns]
    y = data_df["TARGET"]
    
    x = pd.DataFrame(data=x,columns=data_df.columns)
    fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(16,12))
    axs = axs.flatten()
    for i, k in enumerate(data_df.columns):
        sns.regplot(
            y=y,
            x=x[k],
            marker="+", 
            scatter_kws={"color":"blue","alpha":0.3,"s":45},
            line_kws={"color":"red","alpha":1,"lw":4},
            ax=axs[i]
        )
    #更改 Matplotlib 子圖大小和間距
    fig.tight_layout()
    sio = BytesIO()
    plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.3)
    data = base64.encodebytes(sio.getvalue()).decode()
    src = 'data:image/png;base64,' + str(data)
    # 记得关闭，不然画出来的图是重复的
    plt.close()
    
    resp["trend_image"] = src
    return JsonResponse(src,safe=False)

def chart_pearson(resp,data_df):
    sns.set(rc={"figure.figsize":(8,8)})
    sns.heatmap(data=data_df.corr(),cmap="RdBu", #cmap="Greens"
    annot_kws={"size":12},
    annot=True,
    fmt=".2f")
    plt.tight_layout( w_pad=1.0, h_pad=1.0)
    sio = BytesIO()
    plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.3)
    data = base64.encodebytes(sio.getvalue()).decode()
    src = 'data:image/png;base64,' + str(data)
    # 记得关闭，不然画出来的图是重复的
    plt.close()

    resp["pearson_image"] = src
    return JsonResponse(src,safe=False)

#模型訓練
@requires_csrf_token
def train(request):
    post_data = json.loads(request.body)
    dataset_name = post_data["dataset"]
    train_size = post_data["train_size"]
    model_name = post_data["model_name"]
    return JsonResponse(linear_model(dataset_name,train_size,model_name))

def linear_model(dataset_name,train_size,model_name):
    resp={}
    dataset = None
    model = None
    scale = StandardScaler() # Z-scaler 物件，負向不會消失
    if(dataset_name == "房價資料"):
        dataset = boston_data
    elif(dataset_name == "糖尿病資料" ):
        dataset = diabetes_data

    x_train,x_test,y_train,y_test = train_test_split(
        dataset.data,
        dataset.target,
        train_size=train_size,
        random_state=0)
    x_train = scale.fit_transform(x_train)
    x_test = scale.transform(x_test)
    model = None
    if(model_name == "LinearRegression"):
        model = LinearRegression()
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)

        model_predict_chart2(resp,y_test,y_predict)
        model_feature_chart(dataset,resp,model)
        model_score(resp,x_train,x_test,y_train,y_test,model, y_predict )

    elif(model_name  == "PolynomialRegression" ):
        model = make_pipeline(PolynomialFeatures(2),LinearRegression())
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)

        model_predict_chart2(resp,y_test,y_predict)
        # model_feature_chart(resp,model['linearregression'])
        model_score(resp,x_train,x_test,y_train,y_test,model, y_predict )
    elif(model_name == "LassoRegression"):
        model = LassoCV(eps=0.001,n_alphas=100,cv=10)
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)

        model_predict_chart2(resp,y_test,y_predict)
        model_feature_chart(dataset,resp,model)
        model_score(resp,x_train,x_test,y_train,y_test,model, y_predict )
    elif(model_name == "MLP"):
        model  = MLPRegressor(max_iter=500)
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)
        
        model_predict_chart2(resp,y_test,y_predict)
        #model_feature_chart(resp,model)
        model_score(resp,x_train,x_test,y_train,y_test,model, y_predict )
    return resp

#模型分數
def model_score(resp,x_train,x_test,y_train,y_test,model,y_predict):
    #訓練分數
    resp["train_data_r2"] = model.score(x_train,y_train)
    #驗證分數
    resp["validat_data_mse"] =  metrics.mean_squared_error(y_test, y_predict)
    resp["validat_data_rmse"] = np.sqrt(metrics.mean_squared_error(y_test, y_predict))
    resp["validat_data_r2"] = model.score(x_test,y_test) # score 會將x_test拿去做predict

#特徵重要性圖表    
def model_feature_chart(dataset,resp,model):
    #特徵重要性
    plt.figure(figsize=(8,6)) 
    plt.bar(dataset.feature_names,model.coef_)
    plt.tight_layout()
    sio = BytesIO()
    plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.3)
    data = base64.encodebytes(sio.getvalue()).decode()
    plt.close()
    imagebase64 = 'data:image/png;base64,' + str(data)
    resp["coef_chart"] = imagebase64

#預測與實際分布圖
def model_predict_chart(resp,y_test,y_predict):
    plt.figure(figsize=(8,6))
    draw_data = pd.concat([pd.Series(y_test),pd.Series(y_predict)],axis=1)
    draw_data = pd.concat([pd.Series(y_test),pd.Series(y_predict)],axis=1)
    draw_data.columns = ["real","predict"]
   
    sns.lmplot(x="real", y="predict",
        data=draw_data,
        aspect=1.2,
        ci=95)
    plt.tight_layout()
    sio = BytesIO()
    plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.3)
    data = base64.encodebytes(sio.getvalue()).decode()
    plt.close()
    imagebase64 = 'data:image/png;base64,' + str(data)
    resp["predict_chart"] = imagebase64

#預測與實際分布圖2
def model_predict_chart2(resp,y_test,y_predict):
    #用於正常顯示中文，Apple Mac 可選用 Arial
    plt.rcParams["font.sans-serif"] = "Microsoft JhengHei"
    #用於正常顯示符號
    plt.rcParams["axes.unicode_minus"] = False

    #設置圖形大小
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(range(len(y_test)),
    y_test,
    ls="solid",
    lw=1,
    c="b",
    label="真實值")
    plt.plot(range(len(y_predict)),
    y_predict,
    ls="dashdot",
    lw=2,
    c="r",
    label="預測值")
    #繪製網格
    plt.grid(alpha=0.4, linestyle=":")
    plt.legend()
    plt.xlabel("number") #設置 x 軸的標籤文本
    plt.ylabel("房價") #設置 y 軸的標籤文本
    plt.tight_layout()
    sio = BytesIO()
    plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.3)
    data = base64.encodebytes(sio.getvalue()).decode()
    plt.close()
    imagebase64 = 'data:image/png;base64,' + str(data)
    resp["predict_chart"] = imagebase64