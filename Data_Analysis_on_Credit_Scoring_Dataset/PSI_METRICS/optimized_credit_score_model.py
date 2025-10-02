from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np

# Loading Data
df = pd.read_csv("credit_score.csv")

# Preprocessing
df.drop(columns=["CUST_ID"], inplace=True)

# if income = 0 then income = sum of expenditutes + savings
cols = ["T_CLOTHING_12", "T_EDUCATION_12", "T_ENTERTAINMENT_12", "T_EXPENDITURE_12", "T_FINES_12", "T_GAMBLING_12", "T_GROCERIES_12", "T_HEALTH_12", "T_HOUSING_12", "T_TAX_12", "T_TRAVEL_12", "T_UTILITIES_12", "SAVINGS"]
df["calc_inc"] = df[cols].sum(axis=1)
df.loc[df["INCOME"] == 0, "INCOME"] = df["calc_inc"]
df.drop(columns="calc_inc", inplace=True)

# Encoding for Categorical Values
cat = ["High", "Low", "No"]
encoder = OrdinalEncoder(categories=[cat])
df["CAT_GAMBLING"] = encoder.fit_transform(df[["CAT_GAMBLING"]]).astype("int64")

df_psi = df.copy()
scaler = MinMaxScaler()
cols = [col for col in df.columns if df[col].nunique() > 2]
df[cols] = scaler.fit_transform(df[cols])

# Feature Selection
cols = [col for col in df.columns if col != "DEFAULT"]
pca = PCA(n_components=18, random_state=40)
x_pca = pd.DataFrame(pca.fit_transform(df[cols]), columns=[ 'PC-' + str(i) for i in range(1, 19)])

# Train and Test Data
X = x_pca
y = df["DEFAULT"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Model fitting
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)

# Model testing
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# KS satistics
knn.fit(X_train, y_train)
y_pred = knn.predict_proba(X_test)
ks_df = pd.DataFrame({
    "p": np.round(y_pred[:,1], decimals=1),
    "y": y_test
})
ks_df.sort_values(by="p", ascending=False, inplace=True)
ks, p_value = ks_2samp(ks_df.loc[ks_df.y==0,"p"], ks_df.loc[ks_df.y==1,"p"])
print("KS-Stat:", ks, p_value)
ks_df.to_csv("KS.csv")

# Preprocessing
cols = [col for col in df_psi.columns if df_psi[col].nunique() > 2 and col != "CREDIT_SCORE"]
df_psi[cols] = scaler.fit_transform(df_psi[cols])

# Feature Selection
cols = [col for col in df_psi.columns if col != "CREDIT_SCORE"]
pca = PCA(n_components=31, random_state=40)
x_pca = pd.DataFrame(pca.fit_transform(df_psi[cols]), columns=[ 'PC-' + str(i) for i in range(1, 32)])

# Train and Test Data
X = x_pca
y = df_psi["CREDIT_SCORE"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Model fitting
model = LinearRegression()
model.fit(X_train, y_train)

# Model testing
y_pred = model.predict(X_test)
print("R2-Score:", r2_score(y_test, y_pred))

# Binning
def create_binned_data_psi(actual, predicted, n_bins = 10):
    freq_actual, bin_edges = np.histogram(actual, bins=n_bins)
    freq_predicted, bins = np.histogram(predicted, bins=bin_edges)
    bins = []
    for i in range(len(bin_edges) - 1):
        bins.append(str(bin_edges[i]) + " - " + str(bin_edges[i+1]))
    df1 = pd.DataFrame({ "Bins": bins, "Actual_freq": freq_actual, "Predicted_freq": freq_predicted})
    return df1
df1 = create_binned_data_psi(y_test, y_pred)
df2 = pd.concat([pd.DataFrame({"Actual": y_test, "Predicted": y_pred}), df1], axis=0)
df2.to_csv("PSI_METRICS.csv")
    
# PSI Statistics
def psi_stats(data):
    actual_freq = data["Actual_freq"].dropna().to_numpy()
    predicted_freq = data["Predicted_freq"].dropna().to_numpy()
    actual_frac = actual_freq / data["Actual"].dropna().shape[0]
    predicted_frac = predicted_freq / data["Actual"].dropna().shape[0] + 0.001
    diff = actual_frac - predicted_frac
    log = np.log(actual_frac / predicted_frac)
    prod = diff * log
    print(actual_frac)
    print(predicted_frac)
    print(diff)
    print(log)

    return sum(prod)

print("Psi-Stat:", psi_stats(df2))

