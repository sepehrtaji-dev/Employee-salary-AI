import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 2000

data = {
    'Years_Experience': np.random.randint(0, 30, n_samples),
    'Education_Level': np.random.choice([1, 2, 3, 4, 5], n_samples),  # 1=دیپلم, 5=دکترا
    'Age': np.random.randint(22, 60, n_samples),
    'Gender': np.random.choice([0, 1], n_samples),  # 0=زن, 1=مرد
    'City_Type': np.random.choice([0, 1, 2], n_samples),  # 0=روستا, 1=شهر, 2=کلانشهر
    'Hours_Worked': np.random.randint(20, 60, n_samples),
    'Job_Satisfaction': np.random.randint(1, 6, n_samples),  # 1=خیلی کم, 5=خیلی زیاد
    'Department': np.random.choice([0, 1, 2, 3], n_samples),  # 0=فروش, 1=فنی, 2=مالی, 3=مدیریت
    'Salary': np.random.randint(20000000, 150000000, n_samples)
}

df = pd.DataFrame(data)

# اضافه کردن رابطه منطقی بین ویژگی‌ها و حقوق
df['Salary'] = (
    df['Years_Experience'] * 1800000 +
    df['Education_Level'] * 5000000 +
    df['Age'] * 300000 +
    df['Hours_Worked'] * 200000 +
    df['Job_Satisfaction'] * 1000000 +
    df['Department'] * 2000000 +
    np.random.randint(-5000000, 5000000, n_samples)
)

# اطمینان از مثبت بودن حقوق
df['Salary'] = df['Salary'].clip(lower=15000000)

# ذخیره دیتاست
df.to_csv("employee_salary.csv", index=False)
print("✅ دیتاست ساخته شد!")
print(df.head())