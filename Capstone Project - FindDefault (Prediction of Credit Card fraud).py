#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


file_path = '/mnt/data/file-MBs6jeUI0g9KXnbAMiTznTgb'
credit_card_data = pd.read_csv("C:\\Users\\User\\Downloads\\creditcard.csv")


# In[4]:


data_head = credit_card_data.head()
data_info = credit_card_data.info()


# In[5]:


data_head, data_info


# In[6]:


duplicates = credit_card_data.duplicated().sum()


# In[7]:


first_transaction_time = pd.Timestamp('2013-09-01')
credit_card_data['Time_converted'] = credit_card_data['Time'].apply(
    lambda sec: first_transaction_time + pd.Timedelta(seconds=sec))


# In[8]:


statistical_summary = credit_card_data.describe()


# In[9]:


duplicates, statistical_summary


# In[10]:


credit_card_data_unique = credit_card_data.drop_duplicates()


# In[11]:


numerical_columns = credit_card_data_unique.drop(['Time', 'Time_converted', 'Class'], axis=1).columns


# In[12]:


outlier_bounds = credit_card_data_unique[numerical_columns].mean() +                  3 * credit_card_data_unique[numerical_columns].std()


# In[13]:


outliers_count = (credit_card_data_unique[numerical_columns] > outlier_bounds).sum()


# In[14]:


outliers_count, credit_card_data_unique.info()


# In[18]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold


# In[19]:


credit_card_data_unique = credit_card_data.drop_duplicates().copy()
credit_card_data_unique.drop('Time', axis=1, inplace=True)


# In[20]:


robust_scaler = RobustScaler()
credit_card_data_unique['Amount_scaled'] = robust_scaler.fit_transform(
    credit_card_data_unique['Amount'].values.reshape(-1,1))


# In[21]:


credit_card_data_preprocessed = credit_card_data_unique.drop(['Amount'], axis=1)


# In[22]:


X = credit_card_data_preprocessed.drop(['Class'], axis=1)
X.drop(['Time_converted'], axis=1, inplace=True)
y = credit_card_data_preprocessed['Class']


# In[23]:


sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_selected = sel.fit_transform(X)


# In[24]:


smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_selected, y)


# In[25]:


balance_check = y_res.value_counts(normalize=True)


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)


# In[27]:


balance_check, X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[28]:


from sklearn.utils import resample


# In[29]:


data_majority = credit_card_data_preprocessed[credit_card_data_preprocessed.Class == 0]
data_minority = credit_card_data_preprocessed[credit_card_data_preprocessed.Class == 1]


# In[30]:


data_majority_downsampled = resample(data_majority, 
                                     replace=False,
                                     n_samples=len(data_minority),
                                     random_state=42)


# In[31]:


data_balanced = pd.concat([data_majority_downsampled, data_minority])


# In[32]:


balance_check = data_balanced.Class.value_counts()


# In[33]:


X_balanced = data_balanced.drop(['Class', 'Time_converted'], axis=1)
y_balanced = data_balanced['Class']


# In[34]:


X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=42)


# In[35]:


balance_check, X_train_balanced.shape, X_test_balanced.shape, y_train_balanced.shape, y_test_balanced.shape


# In[36]:


credit_card_data_unique['Amount_scaled'] = robust_scaler.fit_transform(
    credit_card_data_unique['Amount'].values.reshape(-1,1))


# In[37]:


credit_card_data_preprocessed = credit_card_data_unique.drop(['Amount'], axis=1)


# In[38]:


X_unbalanced = credit_card_data_preprocessed.drop(['Class', 'Time_converted'], axis=1)
y_unbalanced = credit_card_data_preprocessed['Class']


# In[39]:


data_majority_downsampled = resample(
    credit_card_data_preprocessed[credit_card_data_preprocessed.Class == 0],
    replace=False,
    n_samples=y_unbalanced.sum(),
    random_state=42)


# In[40]:


data_balanced = pd.concat([data_majority_downsampled, credit_card_data_preprocessed[y_unbalanced == 1]])


# In[41]:


X_balanced = data_balanced.drop(['Class', 'Time_converted'], axis=1)
y_balanced = data_balanced['Class']


# In[42]:


sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_balanced = sel.fit_transform(X_balanced)


# In[43]:


X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=42)


# In[44]:


balance_check = y_balanced.value_counts(normalize=True)


# In[45]:


balance_check, X_train_balanced.shape, X_test_balanced.shape, y_train_balanced.shape, y_test_balanced.shape


# In[62]:


robust_scaler = RobustScaler()


# In[63]:


credit_card_data_unique = credit_card_data.drop_duplicates()


# In[65]:


credit_card_data_unique = credit_card_data.drop_duplicates().copy()
credit_card_data_unique['Amount_scaled'] = robust_scaler.fit_transform(
    credit_card_data_unique['Amount'].values.reshape(-1,1))


# In[66]:


credit_card_data_preprocessed = credit_card_data_unique.drop(['Amount'], axis=1)


# In[67]:


class_0 = credit_card_data_preprocessed[credit_card_data_preprocessed['Class'] == 0]
class_1 = credit_card_data_preprocessed[credit_card_data_preprocessed['Class'] == 1]


# In[68]:


class_0_downsampled = resample(class_0, replace=False, n_samples=len(class_1), random_state=42)


# In[69]:


balanced_data = pd.concat([class_0_downsampled, class_1])


# In[70]:


sel = VarianceThreshold(threshold=(.8 * (1 - .8)))


# In[71]:


X_balanced = balanced_data.drop(['Class', 'Time_converted'], axis=1)
X_selected_balanced = sel.fit_transform(X_balanced)


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X_selected_balanced, balanced_data['Class'], test_size=0.3, random_state=42)


# In[73]:


balance_check = pd.Series(y_train).value_counts(normalize=True)


# In[74]:


balance_check, X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[94]:


from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import numpy as np


# In[76]:


robust_scaler = RobustScaler()


# In[77]:


credit_card_data_unique['Amount_scaled'] = robust_scaler.fit_transform(
    credit_card_data_unique['Amount'].values.reshape(-1,1))


# In[78]:


credit_card_data_preprocessed = credit_card_data_unique.drop(['Amount'], axis=1)


# In[79]:


X = credit_card_data_preprocessed.drop(['Class', 'Time_converted'], axis=1)
y = credit_card_data_preprocessed['Class']


# In[80]:


majority_class = credit_card_data_preprocessed[credit_card_data_preprocessed.Class == 0]
minority_class = credit_card_data_preprocessed[credit_card_data_preprocessed.Class == 1]


# In[81]:


majority_downsampled = resample(majority_class, 
                                replace=False,
                                n_samples=len(minority_class),
                                random_state=42)


# In[82]:


balanced_data = pd.concat([majority_downsampled, minority_class])


# In[83]:


balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)


# In[84]:


X_balanced = balanced_data.drop('Class', axis=1)
y_balanced = balanced_data['Class']


# In[85]:


X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=42)


# In[86]:


balance_check = y_balanced.value_counts(normalize=True)


# In[87]:


balance_check, X_train_balanced.shape, X_test_balanced.shape, y_train_balanced.shape, y_test_balanced.shape


# In[88]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


# In[99]:


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)


# In[100]:


X_balanced_numeric = X_balanced.select_dtypes(include=[np.number])


# In[101]:


sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_selected_balanced = sel.fit_transform(X_balanced_numeric)


# In[102]:


X_train_balanced_numeric, X_test_balanced_numeric, y_train_balanced, y_test_balanced = train_test_split(
    X_selected_balanced, y_balanced, test_size=0.3, random_state=42
)


# In[103]:


print(X_train_balanced_numeric.dtype)


# In[104]:


rf_classifier.fit(X_train_balanced_numeric, y_train_balanced)


# In[105]:


y_pred_balanced_numeric = rf_classifier.predict(X_test_balanced_numeric)
y_pred_proba_balanced_numeric = rf_classifier.predict_proba(X_test_balanced_numeric)[:,1]


# In[106]:


roc_auc_balanced_numeric = roc_auc_score(y_test_balanced, y_pred_proba_balanced_numeric)


# In[107]:


class_report_balanced_numeric = classification_report(y_test_balanced, y_pred_balanced_numeric, target_names=['Not Fraud', 'Fraud'])


# In[108]:


roc_auc_balanced_numeric, class_report_balanced_numeric


# In[109]:


X_balanced = balanced_data.drop(['Class', 'Time_converted'], axis=1)


# In[110]:


sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_selected_balanced = sel.fit_transform(X_balanced)


# In[111]:


X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(
    X_selected_balanced, y_balanced, test_size=0.3, random_state=42)


# In[112]:


rf_classifier.fit(X_train_balanced, y_train_balanced)


# In[113]:


y_pred = rf_classifier.predict(X_test_balanced)
y_pred_proba = rf_classifier.predict_proba(X_test_balanced)[:,1]


# In[114]:


roc_auc = roc_auc_score(y_test_balanced, y_pred_proba)


# In[115]:


class_report = classification_report(y_test_balanced, y_pred, target_names=['Not Fraud', 'Fraud'])


# In[116]:


roc_auc, class_report


# In[ ]:




