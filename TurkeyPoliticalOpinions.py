import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.calibration import LabelEncoder

warnings.filterwarnings("ignore",category=UserWarning)
data=pd.read_csv("data.csv")
label_encoder = LabelEncoder()


del data['Timestamp']

data['Bolge'] = label_encoder.fit_transform(data['Bolge'])
data = pd.get_dummies(data, columns=['Bolge'])
data = data.rename(columns={'Bolge_0': 'Akdeniz'})
data = data.rename(columns={'Bolge_1': 'DoğuAnadolu'})
data = data.rename(columns={'Bolge_2': 'Ege'})
data = data.rename(columns={'Bolge_3': 'GüneyDoğu'})
data = data.rename(columns={'Bolge_4': 'Karadeniz'})
data = data.rename(columns={'Bolge_5': 'Marmara'})
data = data.rename(columns={'Bolge_6': 'İçAnadolu'})

data['Yas'] = label_encoder.fit_transform(data['Yas'])
data = pd.get_dummies(data, columns=['Yas'])
data = data.rename(columns={'Yas_0': '0-18'})
data = data.rename(columns={'Yas_1': '18-30'})
data = data.rename(columns={'Yas_2': '30-50'})
data = data.rename(columns={'Yas_3': '50-60'})
data = data.rename(columns={'Yas_4': '60+'})

data[['Akdeniz','DoğuAnadolu','Ege','GüneyDoğu','Karadeniz','Marmara','İçAnadolu']] = data[['Akdeniz','DoğuAnadolu','Ege','GüneyDoğu','Karadeniz','Marmara','İçAnadolu']].astype(int)
data[['0-18','18-30','30-50','50-60','60+']]    =data[['0-18','18-30','30-50','50-60','60+']].astype(int)
for i in range (1,11):
    data[f'soru{i}']=data[f'soru{i}'].replace({'Evet':True,'Hayır':False})

for i in range(1, 11):
    column_name = f'soru{i}' 
    data[column_name] = data[column_name].astype(int)

data['Egitim'] = data['Egitim'].replace({'İlkokul':0,'Ortaokul': 1,'Lise':2,'Ön Lisans':3,'Lisans':4,'Lisans Üstü':5})
data['Cinsiyet']=data['Cinsiyet'].replace({'Erkek':1,'Kadın':0})

data.head(50)


iyiparti=data[data.parti=="IYI PARTI"]
akp=data[data.parti=="AKP"]
chp=data[data.parti=="CHP"]
hdp=data[data.parti=="HDP"]
diger=data[data.parti=="DIĞER"]
plt.hist([iyiparti.Egitim,chp.Egitim,akp.Egitim,hdp.Egitim,diger.Egitim],
         color=["green", "red", "blue", "black","yellow"],
         label=["iyiparti","CHP","AKP","hdp","diger"])
plt.xlabel("Egitim")


plt.legend()

plt.show()



y=data.parti.values
x=data.drop(["parti"],axis=1)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=7)

k_values = list(range(1, 10))
mean_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=5)  # 5x cross value score
    mean_scores.append(scores.mean())

optimal_k = k_values[mean_scores.index(max(mean_scores))]
print("Optimal k value:", optimal_k)

final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
final_knn.fit(x_train, y_train)

accuracy = final_knn.score(x_test, y_test)
print(f"Accuracy with optimal k value: %{accuracy * 100:.2f}")
sc = MinMaxScaler()
sc.fit_transform(x.values)



RFCmodel = RandomForestClassifier()  
RFCmodel.fit(x_train,y_train)
rfc_pred = RFCmodel.predict(x_test)
rfc_acc = accuracy_score(rfc_pred, y_test)
print ("Random forest test accuracy: {:.2f}%".format(rfc_acc*100))
print( "\n" )
print(classification_report(y_test, rfc_pred))
print( "\n" )

def newprediction():
    v1 = float(input("--Sex-- \n[1]=Erkek \n[0]=Kadın\n>> "))
    v2 = int(input("--Age-- \n[0]=0-18 \n[1]=18-30 \n[2]=30-50 \n[3]=50-60 \n[4]=60+\n>> "))
    v3 = int(input("--Areas inhabited in Turkey-- \n[0]=Marmara\n[1]=Akdeniz\n[2]=İç Anadolu\n[3]=Karadeniz\n[4]=Ege\n[5]=Doğu Anadolu\n[6]=Güneydoğu Anadolu \n>> "))
    v4 = float(input("--Education Level--  \n[0]=İlkokul'\n[1]='Ortaokul'\n[2]=Lise\n[3]=Ön Lisans\n[4]=Lisans\n[5]=Lisans Üstü\n>> "))  
    v5 = int(input("Do you think our Economic Status is good?\n[YES]=1 [NO]=0 \n>> "))  
    v6 = int(input("Need Reform in Education?\n[YES]=1 [NO]=0 \n>> "))
    v7 = int(input("Do you support customization?\n[YES]=1 [NO]=0 \n>> "))
    v8 = int(input("Should the state use a penalty like death penalty for certain crimes?\n[YES]=1 [NO]=0 \n>> "))
    v9 = int(input("Do you find our journalists neutral enough?\n[YES]=1 [NO]=0 \n>> "))
    v10 = int(input("From 22:00 am Then Are You Supporting the Prohibition to Buy Drinks?\n[YES]=1 [NO]=0 \n>> "))
    v11 = int(input("Do You Want to Live in a Secular State?\n[YES]=1 [NO]=0 \n>> "))
    v12 = int(input("Are you supporting the abortion ban?\n[YES]=1 [NO]=0 \n>> "))
    v13 = int(input("Do you think that the extraordinary state (Ohal) restricts Freedoms?\n[YES]=1 [NO]=0 \n>> "))
    v14 = int(input("Would you like a new part of the parliament to enter?\n[YES]=1 [NO]=0 \n>> "))
    new_prediction = RFCmodel.predict(np.array([[v1, v2, v3, v4, v5, v6, v7,v8,v9,v10,v11,v12,v13,v14]]))
    print(new_prediction)

while True:
    newprediction()
    choose = input("Do you want to continue? (y/n): ")
    choose = choose.lower()
    if choose == "y":
        continue
    elif choose == "n":
        print("Exiting...")
        break