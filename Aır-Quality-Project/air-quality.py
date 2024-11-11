import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# 1. veri yüklemdimve eksik verileri (-200) NaN yaptım
data = pd.read_csv('air-quality-dataset.csv')
data.replace(-200, np.nan, inplace=True)

# eksik verilerin analizinin yapılma adımı
missing_values_before = data.isnull().sum()

# 2. tarih ve saat bilgilerinden özellik ürettim
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data = data.drop(columns=['Date'])

if 'Time' in data.columns:
    data['Hour'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.hour
    data = data.drop(columns=['Time'])

# 3. korelasyon matrisi oluşturdum
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(15, 12))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Korelasyon Matrisi')
plt.show()

# 4. hedef ve özellikleri tanımladım
targets = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']

# 5. gaz seviyelerinden özellikler ürettim
data['CO_NOx_ratio'] = data['CO(GT)'] / (data['NOx(GT)'] + 1e-5)
data['CO_NO2_ratio'] = data['CO(GT)'] / (data['NO2(GT)'] + 1e-5)
data['Total_Gases'] = data[['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']].sum(axis=1)
data['CO_Hour'] = data['CO(GT)'] * data['Hour']
data['NOx_T'] = data['NOx(GT)'] * data['T']
data['Combined_Interaction'] = data['CO_Hour'] * data['NOx_T']

# ay isimlerini ayarladım
data['Month_Name'] = data['Month'].map({
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
    6: "June", 7: "July", 8: "August", 9: "September",
    10: "October", 11: "November", 12: "December"
})

# 6. Eksik Verileri Doldurma
imputer = SimpleImputer(strategy='median')
numeric_data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)
data_imputed = numeric_data_imputed.copy()
data_imputed['Month_Name'] = data['Month_Name']

# eksik verileri karşılaştırdım (Öncesi/Sonrası)
missing_values_after = data_imputed.isnull().sum()
missing_data_comparison = pd.DataFrame({
    'Feature': missing_values_before[missing_values_before > 0].index,
    'Missing Before': missing_values_before[missing_values_before > 0].values,
    'Missing After': missing_values_after[missing_values_before[missing_values_before > 0].index].values
})
print("Eksik Verileri Doldurma:")
print(missing_data_comparison)

# 7. modelleme için eğitim ve test setleri
X = data_imputed.drop(columns=["CO(GT)", "Month_Name"])
y = data_imputed["CO(GT)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. random forest modeli oluşturdum
param_grid_rf = {
    "n_estimators": [100, 150],
    "max_depth": [10, 15],
    "min_samples_split": [10, 15],
    "min_samples_leaf": [5, 10],
    "max_features": ["sqrt", "log2"]
}
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=1), param_grid_rf, cv=10, scoring="r2", n_jobs=-1)
grid_search_rf.fit(X_train_scaled, y_train)
best_rf_model = grid_search_rf.best_estimator_

# model performansı değerlendirme aşaması:
y_train_pred = best_rf_model.predict(X_train_scaled)
y_test_pred = best_rf_model.predict(X_test_scaled)

train_r2, test_r2 = r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)
train_rmse, test_rmse = mean_squared_error(y_train, y_train_pred, squared=False), mean_squared_error(y_test, y_test_pred, squared=False)

print("Model Performans (R² ve RMSE):")
print(pd.DataFrame({
    "Set": ["Train", "Test"],
    "R² Score": [train_r2, test_r2],
    "RMSE": [train_rmse, test_rmse]
}))

# 9. özelliklerin önemi:
feature_importances = best_rf_model.feature_importances_
indices = np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.title('Özelliklerin Göreceli Önemi')
plt.barh(range(len(indices)), feature_importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Özellik Önem Skoru')
plt.show()

# 10. R² skorları (Her hedef-sensör kombinasyonu için)
r2_scores = pd.DataFrame(index=targets, columns=features)

for target in targets:
    for sensor in features:
        model = LinearRegression()
        model.fit(X_train[[sensor]], y_train)
        y_pred = model.predict(X_test[[sensor]])
        r2_scores.loc[target, sensor] = r2_score(y_test, y_pred)

print("R² Scores for each Target and Sensor Combination:")
print(r2_scores)

# 11. aylık ortalama hava kirliliği grafikleri oluşturdum
def plot_monthly_avg(column, title, color):
    monthly_avg = data.groupby('Month_Name')[column].mean().reindex([
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])
    plt.figure(figsize=(10, 6))
    monthly_avg.plot(kind='bar', color=color)
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(f"Average {column}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_monthly_avg('CO(GT)', "Average Air Pollution (CO) by Month", 'skyblue')
plot_monthly_avg('Total_Gases', "Average Air Pollution (Total Gases) by Month", 'salmon')

# 12. model performansının görselleştirilmesi
def plot_performance(metrics, metric_name):
    plt.figure(figsize=(8, 6))
    plt.bar(metrics["Set"], metrics[metric_name], color=['blue', 'orange'])
    plt.title(f"Train vs Test {metric_name}")
    plt.ylabel(metric_name)
    plt.tight_layout()
    plt.show()

# performans metriklerini gösteren tabloyu yeniden tanımlayarak grafiklerde kullanacağız
performance_metrics = pd.DataFrame({
    "Set": ["Train", "Test"],
    "R² Score": [train_r2, test_r2],
    "RMSE": [train_rmse, test_rmse]
})
print("Model Performans (R² ve RMSE):")
print(performance_metrics)

# grafiklerle gösterdim
plot_performance(performance_metrics, "R² Score")
plot_performance(performance_metrics, "RMSE")

# 13. R² skorlarının ısı haritası
plt.figure(figsize=(10, 8))
sns.heatmap(r2_scores.astype(float), annot=True, cmap='viridis', fmt=".2f")
plt.title("R² Scores for Each Target and Sensor Combination")
plt.xlabel("Sensors")
plt.ylabel("Targets")
plt.tight_layout()
plt.show()