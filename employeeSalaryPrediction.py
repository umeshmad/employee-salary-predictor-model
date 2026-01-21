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
print(f"  R2 Score: {r2:.4f}")

comparison_df=pd.DataFrame({
    'model':list(results.keys()),
    'RMSE':[results[m]['rmse'] for m in results.keys()],
    'MAE':[results[c]['mae'] for c in results.keys()],
    'R2':[results[d]['r2']for d in results.keys()]
}   
)
print(comparison_df.to_string(index=False))
best_model_name=min(results.key(lambda x:results[x]['rmse']))
print(f"the best model name is {best_model_name}")

print("\nGenerating prediction visualizations...")

fig,axes=plt.subplots(1,3,figsize=(18,5))
fig.suptitle('Model Predictions vs Actual Salary',fontsize=16,fontweight='bold')

for index,(name,result) in enumerate(results.items()):
    axes[index].scatter(y_test,results['predictions'],alpha=0.5,color='blue')
    axes[index].plot([y_test.min(),y_test.max()],
                     [y_test.min(),y_test.max()],
                     'r--',lw=2, label='Perfect prediction')
    axes[index].set_xlabel('Actual Salary ($)')
    axes[index].set_ylabel('Predicted Salary ($)')
    axes[index].set_title(f'{name}\nR² = {result["R2"]:.4f}')
    axes[index].legend()
    axes[index].grid(True,alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("FEATURE IMPORTANCE (Random Forest)")
print("=" * 50)

rf_model=results['random']['model']
feature_importance=pd.DataFrame({
    'feature':X.columns,
    'Importance':rf_model.feature_importance_
}).sort_values('Importance',ascending=False)
print(feature_importance.to_string(index=False))

plt.figure(figsize=(10,6))
plt.barh(feature_importance['feature'],feature_importance['Importance'],color='teal')
plt.x_label('Importance')
plt.y_label('Features')
plt.title('Feature Importance in Salary Prediction (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

sample_employees = pd.DataFrame({
    'years_experience': [2, 7, 15],
    'age': [25, 32, 45],
    'education_level': [2, 3, 4],  
    'job_level': [1, 3, 5],  
    'projects_completed': [10, 35, 80],
    'performance_rating': [3, 4, 5]
})

education_labels = {1: 'High School', 2: 'Bachelor', 3: 'Master', 4: 'PhD'}
job_labels = {1: 'Junior', 2: 'Mid', 3: 'Senior', 4: 'Lead', 5: 'Manager'}

print("Sample Employees:")
for i in range(len(sample_employees)):
    print(f"\nEmployee {i+1}:")
    print(f"  Experience: {sample_employees.iloc[i]['years_experience']} years")
    print(f"  Age: {sample_employees.iloc[i]['age']} years")
    print(f"  Education: {education_labels[sample_employees.iloc[i]['education_level']]}")
    print(f"  Job Level: {job_labels[sample_employees.iloc[i]['job_level']]}")
    print(f"  Projects Completed: {sample_employees.iloc[i]['projects_completed']}")
    print(f"  Performance Rating: {sample_employees.iloc[i]['performance_rating']}/5")

best_model = results[best_model_name]['model']
predictions = best_model.predict(sample_employees)

print(f"\nPredicted Salaries (using {best_model_name}):")
for i, salary in enumerate(predictions):
    print(f"  Employee {i+1}: ${salary:,.2f}")











