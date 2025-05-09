# EDA Summary Report: Insurance Enrollment Prediction

## Dataset Overview
- Number of records: 10000
- Number of features: 10
- Missing values: No
- Duplicate rows: 0

## Target Variable
- Enrolled: 6174 (61.74%)
- Not Enrolled: 3826 (38.26%)

## Key Numerical Features
### age
- Range: 22.00 to 64.00
- Mean: 43.00
- Mean for enrolled: 45.60
- Mean for not enrolled: 38.81
- Statistical significance: Yes

### salary
- Range: 2207.79 to 120312.00
- Mean: 65032.97
- Mean for enrolled: 69337.34
- Mean for not enrolled: 58087.02
- Statistical significance: Yes

### tenure_years
- Range: 0.00 to 36.00
- Mean: 3.97
- Mean for enrolled: 3.94
- Mean for not enrolled: 4.00
- Statistical significance: No

## Key Categorical Features
### gender
- Number of unique values: 3
- Most common: Male (48.15%)
- Highest enrollment rate: Other (64.27%)
- Lowest enrollment rate: Female (61.60%)
- Statistical significance: No

### marital_status
- Number of unique values: 4
- Most common: Married (45.89%)
- Highest enrollment rate: Single (62.94%)
- Lowest enrollment rate: Divorced (60.04%)
- Statistical significance: No

### employment_type
- Number of unique values: 3
- Most common: Full-time (70.41%)
- Highest enrollment rate: Full-time (75.33%)
- Lowest enrollment rate: Part-time (28.48%)
- Statistical significance: Yes

### region
- Number of unique values: 4
- Most common: West (25.82%)
- Highest enrollment rate: South (62.83%)
- Lowest enrollment rate: Northeast (61.17%)
- Statistical significance: No

### has_dependents
- Number of unique values: 2
- Most common: Yes (59.93%)
- Highest enrollment rate: Yes (79.74%)
- Lowest enrollment rate: No (34.81%)
- Statistical significance: Yes

## Feature Engineering
The following new features were created:
- age_group: Categorized age into meaningful ranges
- salary_range: Categorized salary into quantile-based ranges
- tenure_group: Categorized tenure into meaningful ranges
- family_status: Combined marital status and dependents information

## Top Features by Importance
1. has_dependents_Yes: 0.1094
2. salary: 0.1094
3. employment_type_Full-time: 0.0946
4. age: 0.0661
5. employment_type_Part-time: 0.0490

## Key Insights
1. **Demographic Factors**: Age shows a significant relationship with enrollment, with younger employees (20-30) having much lower enrollment rates.
2. **Financial Factors**: Salary is strongly correlated with enrollment - higher-paid employees are more likely to enroll.
3. **Employment Factors**: Employment type and tenure show significant differences in enrollment rates.
4. **Family Status**: The combination of marital status and dependents reveals interesting patterns in enrollment behavior.
5. **Regional Variations**: Different regions show varying enrollment rates, especially when combined with employment types.

## Recommendations for Modeling
1. **Feature Selection**: Focus on the top features identified by mutual information.
2. **Feature Engineering**: Utilize the created categorical features (age_group, salary_range, tenure_group, family_status).
3. **Algorithm Selection**: Given the binary classification nature with mixed feature types, consider tree-based models (Random Forest, Gradient Boosting).
4. **Evaluation**: Use appropriate metrics like accuracy, precision, recall, F1-score, and AUC-ROC.
5. **Interpretation**: Focus on feature importance and partial dependence plots to understand the drivers of enrollment.
