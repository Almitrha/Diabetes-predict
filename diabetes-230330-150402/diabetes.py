# Feature Engineering


# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin
# edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli
# geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını
# gerçekleştirmeniz beklenmektedir.


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_ = pd.read_csv("diabetes/diabetes.csv")

df = df_.copy()

df.head()

#Pregnancies: Hamilelik sayısı
#Glucose: Kan şekeri ölçümü
#BloodPressure: Kan basıncı ölçümü
#SkinThickness: Cilt kalınlığı ölçümü
#Insulin: İnsülin ölçümü
#BMI: Vücut kitle indeksi
#DiabetesPedigreeFunction: Diyabet soy ağacı fonksiyonu (kalıtımsal risk tahmini)
#Age: Yaş
#Outcome: Diyabet olup olmadığı (1: evet, 0: hayır)


df.describe().T
df.shape




#Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)
# Numerik değişkenlere göre hedef değişkenin ortalamalarını hesaplayın

for var in num_cols:
    print(df.groupby("Outcome")[var].mean())

"""
Outcome
0   3.298
1   4.866
Name: Pregnancies, dtype: float64
Outcome
0   109.980
1   141.257
Name: Glucose, dtype: float64
Outcome
0   68.184
1   70.825
Name: BloodPressure, dtype: float64
Outcome
0   19.664
1   22.164
Name: SkinThickness, dtype: float64
Outcome
0    68.792
1   100.336
Name: Insulin, dtype: float64
Outcome
0   30.304
1   35.143
Name: BMI, dtype: float64
Outcome
0   0.430
1   0.550
Name: DiabetesPedigreeFunction, dtype: float64
Outcome
0   31.190
1   37.067
"""
#Adım 5: Aykırı gözlem analizi yapınız.
def outlier_thresholds(dataframe, col_name, q1=0.5, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(check_outlier(df, col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

""""
['Pregnancies',
 'Glucose',
 'BloodPressure',
 'SkinThickness',
 'Insulin',
 'BMI',
 'DiabetesPedigreeFunction',
 'Age']
 False
True
True
True
True
True
True
False
"""
#Adım 6: Eksik gözlem analizi yapınız.
df.isnull().values.any()
#False

#Adım 7: Korelasyon analizi yapınız.
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="YlGnBu")
plt.show(block=True)

sns.boxplot(x=df["DiabetesPedigreeFunction"])
plt.show(block=True)

#Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
#değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
#olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
#değerlere işlemleri uygulayabilirsiniz.


# Glikoz, Kan Basıncı, Deri Kalınlığı, Insülin, BMI değişkenlerindeki 0 değerlerini NaN ile değiştirin
df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.nan)

# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()
# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

missing_values_table(df, True)

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

# değişkenlerin standartlatırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


# knn'in uygulanması.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
msno.bar(df)
plt.show(block=True)

msno.matrix(df)
plt.show(block=True)

msno.heatmap(df)
plt.show(block=True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

na_cols = missing_values_table(df, True)
missing_vs_target(df, "Outcome", na_cols)

"""                 TARGET_MEAN  Count
Glucose_NA_FLAG                    
0                      0.349    763
1                      0.400      5
                       TARGET_MEAN  Count
BloodPressure_NA_FLAG                    
0                            0.344    733
1                            0.457     35
                       TARGET_MEAN  Count
SkinThickness_NA_FLAG                    
0                            0.333    541
1                            0.388    227
                 TARGET_MEAN  Count
Insulin_NA_FLAG                    
0                      0.330    394
1                      0.369    374
             TARGET_MEAN  Count
BMI_NA_FLAG                    
0                  0.351    757
1                  0.182     11
"""

#Adım 2: Yeni değişkenler oluşturunuz.

dff['BMI_Age'] = dff['BMI'] * dff['Age']
dff['Age_Category'] = pd.cut(dff['Age'], bins=[0, 20, 40, 60, np.inf], labels=['0-20', '20-40', '40-60', '60+'])
bins = [0, 70, 100, 125, np.inf]
labels = ['Low', 'Normal', 'Pre-diabetic', 'Diabetic']
dff['Glucose_Category'] = pd.cut(dff['Glucose'], bins=bins, labels=labels)

dff.head()

#Adım 3: Encoding işlemlerini gerçekleştiriniz.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

#############################################
# One-Hot Encoding
#############################################

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in dff.columns if 10 >= dff[col].nunique() > 2]

dff = one_hot_encoder(dff, ohe_cols)

dff.head()

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

for col in cat_cols:
    cat_summary(dff, col)

"""       Outcome  Ratio
0.000      500 65.104
1.000      268 34.896
##########################################
   Age_Category_20-40  Ratio
1                 574 74.740
0                 194 25.260
##########################################
   Age_Category_40-60  Ratio
0                 601 78.255
1                 167 21.745
##########################################
   Age_Category_60+  Ratio
0               741 96.484
1                27  3.516
##########################################
   Glucose_Category_Normal  Ratio
0                      567 73.828
1                      201 26.172
##########################################
   Glucose_Category_Pre-diabetic  Ratio
0                            511 66.536
1                            257 33.464
##########################################
   Glucose_Category_Diabetic  Ratio
0                        469 61.068
1                        299 38.932
##########################################
"""

def rare_analyser(dataframe, target, cat_cols):
        for col in cat_cols:
            print(col, ":", len(dataframe[col].value_counts()))
            print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                                "RATIO": dataframe[col].value_counts() / len(dataframe),
                                "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(dff, "Outcome", cat_cols)

"""
Outcome : 2
       COUNT  RATIO  TARGET_MEAN
0.000    500  0.651        0.000
1.000    268  0.349        1.000
Age_Category_20-40 : 2
   COUNT  RATIO  TARGET_MEAN
0    194  0.253        0.526
1    574  0.747        0.289
Age_Category_40-60 : 2
   COUNT  RATIO  TARGET_MEAN
0    601  0.783        0.288
1    167  0.217        0.569
Age_Category_60+ : 2
   COUNT  RATIO  TARGET_MEAN
0    741  0.965        0.352
1     27  0.035        0.259
Glucose_Category_Normal : 2
   COUNT  RATIO  TARGET_MEAN
0    567  0.738        0.441
1    201  0.262        0.090
Glucose_Category_Pre-diabetic : 2
   COUNT  RATIO  TARGET_MEAN
0    511  0.665        0.384
1    257  0.335        0.280
Glucose_Category_Diabetic : 2
   COUNT  RATIO  TARGET_MEAN
0    469  0.611        0.192
1    299  0.389        0.595
"""

scaler = StandardScaler()
dff[num_cols] = scaler.fit_transform(dff[num_cols])

dff[num_cols].head()

#############################################
# Model
#############################################

y = dff["Outcome"]
X = dff.drop(["Outcome"], axis=1)

from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=25)
rf_model = RandomForestClassifier(random_state=45).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
# Out[59]: 0.8095238095238095

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)