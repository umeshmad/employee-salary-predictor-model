import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_Squared_error,r2_score, mean_absolute_error

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
    df['education_leve']*800+
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

fig,axes=plt.subplot(2,3,figsize=(15,10))
fig.suptitle('Employee Salary Analysis',fontsize=16, fontweight='bold')

axes[0,0].hist(df['salary'],bins=30,color='Skyblue',edgecolor='black')
axes[0,0].set_Xlabel('salary $')
axes[0,0].set_Ylabel('frequancy')
axes[0,0].set_title('salary Distribution')

axes[0,1].scatter(df['Years_Expiriance'],df['salary'],alpha=0.5,color="green")
axes[0,1].set_Xlabel("Years of Experiance")
axes[0,1].set_Ylabel("Salary")
axes[0,1].set_title("Experience vs Salary")

axes[0,2].boxplot([df[df['education_level']==i]['salary'].values for i in range(1,5)],
                  labels=['HS','Bacholer','Master','phd'])
axes[0,2].set_Xlabel('education_level')
axes[0,2].set_Ylabel('Salary')
axes[0,2].set_title("Salary by Education Level")

axes[1,0].boxplot([df[df['job_Level']==i]['salary'].values for i in range(1,5)],
                  labels=['Junior', 'Mid', 'Senior', 'Lead', 'Manager'])
axes[1,0].set_Xlabel('job Level')
axes[1,0].set_Ylabel('salary')
axes[1,0].set_title('Salary by job Level')

axes[1,1].scatter(df['prefromance_rating'],df['salary'],alpha=0.5,color="purple")
axes[1,1].set_Xlabel('Performance Rating (1-5)')
axes[1,1].set_Ylabel('salary')
axes[1,1].set_title("Performance vs Salary")









