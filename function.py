import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def plotset(figsize = (13,8), title= None):
    """
    Plot setting function
    """
    plt.figure(figsize= figsize)
    plt.title(title)
    return


def df_binning(Dataframe, column, bins= 10, new_column_name= 'BinnedDiscount'):
    """
    Creates bins in a dataframe and those 
    """
    hist, bin_edge = np.histogram(Dataframe[column], bins= bins, density= True)
    Dataframe[new_column_name] = pd.cut(Dataframe[column], bin_edge)
    return bin_edge

def df_bootstrapping(Dataframe, value_column, bin_column, num_of_samples=100):
    aa = []
    bb = []
    for i in Dataframe[bin_column].value_counts().index:
        a = bootstrapping(Dataframe.loc[Dataframe[bin_column] == i].Quantity, num_of_samples=num_of_samples)
        
        for j in a:
            aa.append(j)
            bb.append(i)
    df_means = pd.DataFrame(aa, columns=[value_column])
    df_means['BinnedValue'] = bb
    return df_means

def welch_t(a, b):
    
    """ Calculate Welch's t statistic for two samples. """
    
    #Convert to numpy arrays in case lists are passed in
    a = np.array(a)
    b = np.array(b)

    numerator = a.mean() - b.mean()

    # “ddof = Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
    #  where N represents the number of elements. By default ddof is zero.

    denominator = np.sqrt(a.var(ddof=1)/a.size + b.var(ddof=1)/b.size)

    return np.abs(numerator/denominator)

import sqlite3


class MyConn():


    def __init__(self, filename=None):
        self.conn = sqlite3.connect(filename)
        self.cursor = self.conn.cursor()
        pass


    def list_tables(self):
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        results = self.cursor.execute(query).fetchall()
        for r in results:
            print(r[0])
        pass


    def build_select_all_query(self, table_name=None):
        try:
            query = "select * from {}".format(self.table_name)
        except Exception as e:
            print("---no tablename in object, exception:---\n{}".format(e))
            query = "select * from {}".format(table_name)
        return query

    
    def get_table_description(self, table_name=None):
        try:
            query = 'select * from {}'.format(self.table_name)
        except:
            query = 'select * from {}'.format(table_name)
        self.cursor.execute(query)
        return [x[0] for x in self.cursor.description]
    

    def load_table_as_df(self, table_name):
        """
        Loads a table of your sqlite db into a pandas df
        Input
        table_name: str, name of your table
        
        Return
        df: pandas dataframe
        """
        query = self.build_select_all_query(table_name=table_name)
        df = pd.read_sql(query, self.conn)
        return df

    
    def load_query_as_df(self, query):
        df = pd.read_sql(query, self.conn)
        return df
    

