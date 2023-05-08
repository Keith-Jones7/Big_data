import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc("font", family='KaiTi', size='12')
data = pd.read_csv("E://Study//研1下//4大数据分析与应用//Homework//Work 1//Credit.csv")

# 将分类变量进行数值化处理

# 1. 处理性别数据  男性为1，女性为2，其他为3
data.loc[data['GENDER'] == "Male", 'GENDER'] = 2
data.loc[data['GENDER'] == "Female", 'GENDER'] = 1
data.loc[(data['GENDER'] != 1) & (data['GENDER'] != 2), 'GENDER'] = 3

# 2. 处理婚姻状况数据，已婚为1，未婚为2，离异为3，未知为4
data.loc[data['MARITAL_STATUS'] == "Married", 'MARITAL_STATUS'] = 1
data.loc[data['MARITAL_STATUS'] == 'Single', 'MARITAL_STATUS'] = 2
data.loc[data['MARITAL_STATUS'] == "Divorce", 'MARITAL_STATUS'] = 3
data.loc[data['MARITAL_STATUS'] == "Unknown", 'MARITAL_STATUS'] = 4

# 3. 处理借贷类型数据，一手为1，二手为2，如有其他为3
data.loc[data['LOANTYPE'] == "Frist-Hand", 'LOANTYPE'] = 1
data.loc[data['LOANTYPE'] == "Second-Hand", 'LOANTYPE'] = 2
data.loc[(data['LOANTYPE'] != 1) & (data['LOANTYPE'] != 2), 'LOANTYPE'] = 3

# 4. 处理支付类型数据，A为1， M为2，其他为3
data.loc[data['PAYMENT_TYPE'] == "Average_Capital_Plus_Interest_Repayment", 'PAYMENT_TYPE'] = 1
data.loc[data['PAYMENT_TYPE'] == "Matching_The_Principal_Repayment", 'PAYMENT_TYPE'] = 2
data.loc[(data['PAYMENT_TYPE'] != 1) & (data['PAYMENT_TYPE'] != 2), 'PAYMENT_TYPE'] = 3

# 处理异常和缺失数据

# 处理月薪数据
plt.figure()
plt.title(label="未处理的月薪数据")
plt.plot(data['MONTHLY_INCOME_WHITHOUT_TAX'])

data.loc[data['MONTHLY_INCOME_WHITHOUT_TAX'] > 800000, 'MONTHLY_INCOME_WHITHOUT_TAX'] = -1
sub_data = data.loc[data['MONTHLY_INCOME_WHITHOUT_TAX'] >= 0, 'MONTHLY_INCOME_WHITHOUT_TAX']
data['MONTHLY_INCOME_WHITHOUT_TAX'].fillna(np.mean(sub_data), inplace=True)
plt.figure()
plt.title(label="处理后的月薪数据")
plt.plot(data['MONTHLY_INCOME_WHITHOUT_TAX'])

# 处理抵押物的总价值数据
plt.figure()
plt.title(label="未处理的抵押物数据")
plt.plot(data['GAGE_TOTLE_PRICE'])
data['GAGE_TOTLE_PRICE'].fillna(np.mean(data['GAGE_TOTLE_PRICE']))

# 处理申请贷款数据
plt.figure()
plt.title(label="未处理的申请贷款数据")
plt.plot(data['APPLY_AMOUNT'])
data['APPLY_AMOUNT'].fillna(np.mean(data['APPLY_AMOUNT']))

# 处理申请期限数据
plt.figure()
plt.title("未处理的申请期限数据")
plt.plot(data['APPLY_TERM_TIME'])

# 处理申请利率数据
plt.figure()
plt.title("未处理的申请利率数据")
plt.plot(data['APPLY_INTEREST_RATE'])

output = "E://Study//研1下//4大数据分析与应用//Homework//Work 1//Credit_handle.csv"
data.to_csv(output, sep=',', index=False)
