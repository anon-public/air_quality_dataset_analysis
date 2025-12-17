import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv("Air quality dataset.csv")
df.head(10)



des_stat = df["aqi_log"].agg(["mean","median","std","count"])
print("\nDescriptive Statistics:\n",des_stat)
z_score = stats.zscore(df["aqi_log"])
ot = df[np.abs(z_score>3)]
print("\nOutliers:",ot)

mean_val = des_stat["mean"]
std_val = des_stat["std"]
prob = stats.norm.cdf(2.0,loc=mean_val,scale=std_val)
print(f"Probability > 2: {prob:.4f}")


SE = mean_val/(np.sqrt(std_val))
c_interval = stats.norm.interval(0.95,loc=mean_val,scale=SE)
print("95% Confidence interval:",c_interval)


#One sample t test ,benchmark =1.5 AQI Log
hyp_mean= 1.5
t_stat,p_val=stats.ttest_1samp(df["aqi_log"],popmean=hyp_mean)
print("One Sample T test result")
print("-----------------------")
print(f"Null hypothesis(H0):{hyp_mean:.4f}")
print(f"Sample Mean: {df['aqi_log'].mean():.4f}")
print(f"t Statistic:{t_stat:.4f}")
print(f"p value:{p_val:.4f}")

if p_val < 0.05:
    print("Reject the null hypothesis")
else:
    print("Fail to redject the null hypotesis")
#concluding the average log‑AQI is significantly higher than the benchmark, indicating air quality is, on average, worse than that target.



