import numpy as np
import pandas as pd
# Normality Test- Sharpio- welk
import scipy.stats as scs
print('ip')
def shapiro_normal(DataSeries):
    """
    Shapiro Welk test normality.  
    Input is single dataseries, array, or list.  
    Output- t statistic, P statistic
    The null hypothesis is that the tested distribution is normally distributed, and the alternative hypothesis is that the distribution is non-normal. A p values threshold is 0.05.
    """
    t, p = scs.shapiro(DataSeries)
    if p< 0.05:
        print('non normal')
    else:
        print('normal')
    return f"t {t}, p {p}"

def bootstrapping(DataSeries, num_of_samples=100):
    """means of randomly selected collection from the given dataset. Selection is done with replacement, 
    and the collection is the same size of the dataset.  
    Input is single dataseries, array, or list."""
    return [np.random.choice(DataSeries, size=DataSeries.shape[0], replace=True).mean() for i in range(num_of_samples)]

def levene_variances(Data1, Data2):
    """
    h0: var_x1 = var_x2
    ha: var_x1 != var_x2
    """
    t, p = scs.levene(Data1, Data2)
    if p < 0.05:
        print(f"p = {p}\nTherefore the data do not have equal variances")
        return False
    print(f"p = {p}\nTherefore the data have equal variances")
    return True

def cohen_d(Data1, Data2):
    """
    
    """
    nx= len(Data1)
    ny= len(Data2)
    dof = nx + ny - 2
    std = np.sqrt(((nx-1)*np.std(Data1, ddof=1) ** 2 + (ny-1)*np.std(Data2, ddof=1) ** 2) / dof)
    return (np.mean(Data1)- np.mean(Data2))/ std

def sql_to_dataframe(query):
    cur.execute(query)
    
    df = pd.DataFrame(cur.fetchall()) #Take results and create dataframe
    df.columns = [i[0] for i in cur.description]

    return df