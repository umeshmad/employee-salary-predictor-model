import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

np.random.seed(42)
n_samples=400
data={
    'Years_Expiriance':np.random.randint(0,20, n_samples),
    'age':np.random.randint(20,61, n_samples),
    'education_level':np.random.randint(1,5,n_samples),
    'job_Level':np.random.randint(1,6,n_samples),
    'project_Completed':np.random.randint(5,101,n_samples),
    'prefromance_rating':np.random.randint(1,6,n_samples)
}
df=pd.DataFrame(data)
df['salary']=(
    df['Years_Expiriance']*3000+
    df['age']*500+
    df['education_level']*800+
    df['job_Level']*12000+
    df['project_Completed']*100+
    df['prefromance_rating']*500+
    np.random.normal(0,5000,n_samples)
)
df['salary']=df['salary']+30000
df['salary']=df['salary'].clip(lower=25000,upper=200000)
print(f"dataset created with{len(df)} employees \n")
print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print(df.head(10))
print("\n"+"="*50)
print("STATISTICAL SUMMARY")
print(df.describe())
print("\n"+"="*50)
print("data info")
print(df.info())
print("\n"+"="*50)
print("Checking the mising values")
print("\n"+"="*50)
print(df.isnull().sum())

print("\nGenerating visualizations...")

fig,axes=plt.subplots(2,3,figsize=(15,10))
fig.suptitle('Employee Salary Analysis',fontsize=16, fontweight='bold')

axes[0,0].hist(df['salary'],bins=30,color='Skyblue',edgecolor='black')
axes[0,0].set_xlabel('Salary $')
axes[0,0].set_ylabel('frequancy')
axes[0,0].set_title('salary Distribution')

axes[0,1].scatter(df['Years_Expiriance'],df['salary'],alpha=0.5,color="green")
axes[0,1].set_xlabel("Years of Experiance")
axes[0,1].set_ylabel("Salary")
axes[0,1].set_title("Experience vs Salary")

axes[0,2].boxplot([df[df['education_level']==i]['salary'].values for i in range(1,5)],
                  labels=['HS','Bacholer','Master','phd'])
axes[0,2].set_xlabel('education_level')
axes[0,2].set_ylabel('Salary')
axes[0,2].set_title("Salary by Education Level")

axes[1,0].boxplot([df[df['job_Level']==i]['salary'].values for i in range(1,5)],
                  labels=['Junior', 'Mid', 'Senior', 'Manager'])
axes[1,0].set_xlabel('job Level')
axes[1,0].set_ylabel('salary')
axes[1,0].set_title('Salary by job Level')

axes[1,1].scatter(df['prefromance_rating'],df['salary'],alpha=0.5,color="purple")
axes[1,1].set_xlabel('Performance Rating (1-5)')
axes[1,1].set_ylabel('salary')
axes[1,1].set_title("Performance vs Salary")

correlation_matrix=df.corr()
sns.heatmap(correlation_matrix,annot=True,fmt='.2f', cmap='coolwarm',
            ax=axes[1,2],cbar_kws={'label':'Correlation'})
axes[1, 2].set_title('Feature Correlation')

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("PREPARING DATA FOR TRAINING")
print("=" * 50)

X=df.drop('salary',axes=1)
y=df['salary']

X_train, X_test, y_train, y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
print(f"Training test size {len(X_train)} samples")
print(f"Training test size {len(y_test)} samples")
print(f"\n Features Being Used:{list(X.columns)}")

print("\n" + "=" * 50)
print("TRAINING MODELS")
print("=" * 50)

models={
    'Linear Regression':LinearRegression(),
    'Decision tree':DecisionTreeClassifier(max_depth=10,random_state=42),
    'Random Forest':RandomForestClassifier(max_depth=10,n_estimators=100,random_state=42)
}
results={}
for name, model in models.items():
    print(f'\nTraning{name}..')
    model.fit(X_train,y_train)

y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)

results[name]={
    'model':model,
    'predictions':y_pred,
    'rmse':rmse,
    'mae':mae,
    'r2':r2
}
print(f"  RMSE: ${rmse:,.2f}")
print(f"  MAE: ${mae:,.2f}")
print(f"  R² Score: {r2:.4f}")











