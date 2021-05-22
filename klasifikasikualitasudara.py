import seaborn as sns

import pandas as pd
import numpy as np
import streamlit as st
#from openpyxl import load_workbook

from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Klasifikasi Kualitas Udara di Provinsi Riau menggunakan Metode Support Vector Machine')
st.write(' ')
st.write(' ')

uploadIspu = st.sidebar.file_uploader("Upload Data Ispu", type=['xlsx'], key = 'ispu')
uploadCuaca = st.sidebar.file_uploader("Upload Data Cuaca", type=['xlsx'], key = 'cuaca')

split_data = st.sidebar.slider('Split Data Training dan Testing', 0.10, 0.30)

if uploadIspu and uploadCuaca is not None:
	ispu = pd.read_excel(uploadIspu, engine='openpyxl', keep_default_na=False, na_values=[""])
	DataSelected = False

	cuaca = pd.read_excel(uploadCuaca, engine='openpyxl', keep_default_na=False, na_values=[""])
	DataSelected = False
	st.write('**1. Data Kualitas Udara**')
	st.write(ispu)
	st.write(ispu.dtypes)
	st.write(' ')

	st.write('**2. Data Cuaca**')
	st.write(cuaca)
	st.write(cuaca.dtypes)

	st.write('Jumlah data ISPU sebelum preprocessing')
	st.write(ispu.shape)
	ispu_dup = ispu[ispu.duplicated(subset=['TANGGAL','ISPU_PM10'], keep=False)]
	st.write("duplikat tgl dan ispu_pm10: " , ispu_dup.shape)
	st.write(ispu_dup)

	st.write('Jumlah data cuaca sebelum preprocessing')
	st.write(cuaca.shape)
	cuaca_dup = cuaca[cuaca.duplicated(subset=['TANGGAL'], keep=False)]
	st.write("duplikat tgl : ", cuaca_dup.shape)
	st.write(cuaca_dup)

	cuaca_dup_measurement = cuaca[cuaca.duplicated(subset=['TANGGAL','Tn','Tx'], keep=False)]
	st.write("duplikat tgl, tn, tx: ", cuaca_dup_measurement.shape)
	st.write(cuaca_dup_measurement)

	st.write('**3. Preprocessing**')
	st.write(' ')
	st.write('**Selection dan Cleaning**')
	st.write('Data Selection dilakukan penghapusan atribut yang tidak diperlukan pada data ispu dan cuaca') 
	st.write('Data Cleaning dilakukan penggantian data tidak relevan (kosong, 8888) dengan rata-rata tiap atribut')
	if DataSelected:
		st.write("Data sudah terseleksi, tidak ada yang perlu di drop") 
	else:
		unused = ['KODE STASIUN','ALAMAT','PROPINSI','KAB/KOTA','JAM','SO2', 'CO', 'O3', 'NO2','ISPU_SO2', 'ISPU_CO', 'ISPU_O3', 'ISPU_NO2']

		ispu.drop(columns=unused, inplace=True)
		st.write('Hasil Selection data ISPU')
		st.write(ispu.shape)

		st.write('mean dari ispu')
		st.write(ispu.mean())
		ispu.fillna(ispu.mean(), inplace=True)
		st.write('Mengecek ada data yang kosong atau tidak pada data ISPU')
		st.write(ispu.isnull().any())
		st.write('Mengecek ada data 0 atau tidak')
		st.write(ispu.apply(lambda x : x == 0).sum())
		ispu.replace(0, ispu.mean(), inplace=True)
		st.write('Hasil Cleaning data ISPU')
		st.write(ispu)
		st.write(ispu.shape)

		unused = ['KAB/KOTA', 'ddd_car', 'ddd_x']

		cuaca.drop(columns=unused, inplace=True)
		st.write('Hasil Selection data Cuaca')
		st.write(cuaca.shape)

		st.write('Mengecek ada data 8888 atau tidak')
		st.write(cuaca.apply(lambda x : x == 8888).sum())
		cuaca.replace(8888, np.nan, inplace=True)
		st.write('mean dari cuaca')
		st.write(cuaca.mean())
		st.write('Mengecek ada data 0 atau tidak')
		st.write(cuaca.apply(lambda x : x == 0).sum())
		cuaca.replace(0, cuaca.mean(), inplace=True)
		st.write('Mengecek ada data yang kosong atau tidak pada data cuaca')
		st.write(cuaca.isnull().any())
		cuaca.fillna(cuaca.mean(), inplace=True)
		st.write('Hasil Cleaning data cuaca')
		st.write(cuaca)
		st.write(cuaca.shape)

		st.write("Hasil Cleaning dan Selection data ISPU : ")
		st.write(ispu)
		st.write(ispu.shape)
		st.write("Hasil Cleaning dan Selection data Cuaca : ")
		st.write(cuaca)
		st.write(cuaca.shape)
		DataSelected = True

	st.write('Mendeteksi nilai yang mengandung 8888')
	st.write('Nilai 8888 merupakan data yang tidak terukur')
	st.write(cuaca.apply(lambda x : x == 8888).sum())

	st.write('**Transformation**')
	st.write('Menampilkan tipe data ISPU sebelum dilakukan transformasi')
	st.write(ispu.dtypes)
	st.write(ispu.shape)
	st.write('Menampilkan tipe data Cuaca sebelum dilakukan transformasi')
	st.write(cuaca.dtypes)
	st.write(cuaca.shape)
	ispu['TANGGAL']=pd.to_datetime(ispu.TANGGAL, format='%Y/%m/%d') 
	cuaca['TANGGAL']=pd.to_datetime(cuaca.TANGGAL, format='%Y/%m/%d') 
	
	st.write('Perubahan type data ISPU (TANGGAL) menjadi datetime64')
	st.write(ispu.dtypes)
	st.write(ispu.shape)
	st.write('Perubahan type data Cuaca (TANGGAL) menjadi datetime64')
	st.write(cuaca.dtypes)
	st.write(cuaca.shape)

	st.write('Transformation data ISPU')
	st.write('Data ISPU ada duplikasi tanggal. Maka dilakukan pengelompokan TANGGAL lalu agregasi Klasifikasi (max), PM10 (mean) dan ISPU_PM10 (mean)')
	
	ispu_new = ispu.groupby( by='TANGGAL').agg({'Klasifikasi':np.max,'PM10': np.mean, 'ISPU_PM10': np.mean})
	st.write(ispu_new)
	st.write(ispu_new.shape)

	st.write('Transformation data Cuaca')
	st.write('Pengelompokan tanggal lalu dilakukan agregasi dengan rata-rata pada setiap atribut')
	st.write('mean dari cuaca')
	st.write(cuaca.mean())
	cuaca_new = cuaca.groupby( by='TANGGAL').mean()
	st.write(cuaca_new)
	st.write(cuaca_new.shape)

	st.write('**Integration**')
	st.write('Menggabungkan antara ispu_new dengan cuaca_new')
	gab = pd.merge(left=ispu_new, right=cuaca_new, how='inner', left_on=['TANGGAL'], right_on=['TANGGAL'])
	st.write(gab)
	st.write('Jumlah data integration')
	st.write(gab.shape)

	st.write('Mengecek ada data yang kosong atau tidak')
	st.write(gab.isnull().any())

	st.write('**Correlation Heatmap**')
	relasi = plt.figure(figsize=(12,10))
	ax = sns.heatmap(gab.corr(), annot=True, cmap='viridis')
	bottom, top = ax.get_ylim()
	ax.set_ylim(bottom + 0.5, top - 0.5)
	st.pyplot(relasi)

	st.write('**Chart Atribut Klasifikasi**')
	st.write(gab['Klasifikasi'].value_counts())

	obj1 = gab.apply(lambda x: True if x['Klasifikasi'] == 1 else False, axis=1)
	obj2 = gab.apply(lambda x: True if x['Klasifikasi'] == 2 else False, axis=1)
	obj3 = gab.apply(lambda x: True if x['Klasifikasi'] == 3 else False, axis=1)
	obj5 = gab.apply(lambda x: True if x['Klasifikasi'] == 5 else False, axis=1)

	jumlahObj1 = len(obj1[obj1 == True].index)
	jumlahObj2 = len(obj2[obj2 == True].index)
	jumlahObj3 = len(obj3[obj3 == True].index)
	jumlahObj5 = len(obj5[obj5 == True].index)

	Data = {"Klasifikasi": [jumlahObj1,jumlahObj2,jumlahObj3,jumlahObj5]}
	df=pd.DataFrame(Data, columns=["Klasifikasi"], index = ['1','2','3','5'])
	df.plot.pie(y="Klasifikasi", figsize=(10, 6), autopct='%1.1f%%', startangle=90)
	bayesChart = plt.show()
	st.pyplot(bayesChart)

	st.write('**4. Algoritma SVM**')
	st.write('**Training dan Testing**')
	X = gab.drop(['Klasifikasi'], axis=1)
	y = gab['Klasifikasi']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_data, stratify=y, random_state=0)
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.fit_transform(X_test)
	st.write('Jumlah Data Testing')
	st.write(len(X_test_scaled))

	st.write('Jumlah Data Training')
	st.write(len(X_train_scaled))

	st.write('Parameter yang digunakan')
	param_grid = [
              {'C': [100], 'gamma':['scale'] ,'kernel': ['linear', 'rbf' , 'poly']}
             ]
	st.write(param_grid)

	grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, n_jobs=3, cv=5, iid =False)
	grid.fit(X_train_scaled,y_train)

	st.write('**Hasil estimator terbaik**')
	st.write(grid.best_estimator_)

	#Penggunaan hasil estimator terbaik
	svclassifier = SVC(C=100, cache_size=200, class_weight='balanced', coef0=0.0,
						decision_function_shape='ovr', degree=3, gamma='scale', kernel='poly',
						max_iter=-1, probability=False, random_state=1, shrinking=True,
						tol=0.001, verbose=False)
	
	svclassifier.fit(X_train_scaled, y_train)

	st.write('**5. Pengujian**')
	y_pred  = svclassifier.predict(X_test_scaled)
	cm = plt.figure(figsize=(5,4))
	dx = sns.heatmap(confusion_matrix(y_test,y_pred), annot=True,cmap='Blues', fmt='g')
	bottom, top = dx.get_ylim()
	dx.set_ylim(bottom + 0.5, top - 0.5)
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	st.pyplot(cm)

	st.write('**Confusion Matrix**')
	st.write(classification_report(y_test,y_pred))

	st.write('**Hasil training dan testing**')
	st.write("Hasil skor Training SVM: %f" % svclassifier.score(X_train_scaled , y_train))
	st.write("Hasil skor Testing SVM: %f" % svclassifier.score(X_test_scaled  , y_test ))


else:
	st.write('Upload Data ISPU dan Cuaca')
