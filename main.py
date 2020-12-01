import pandas as pd
from sklearn.metrics import f1_score

def compute_f1():
    df = pd.read_csv('test.psv', sep='|', header=1)
    df['dates'] = pd.to_datetime(df['dates'])
    df['daysofweek'] = df['dates'].dt.day_name()
    data = df[df['daysofweek'] == 'Thursday'].values
    y_true = data[:, 1].tolist()
    y_pred = data[:, 2].tolist()
    return f1_score(y_true, y_pred)


if __name__ == '__main__':
    print(compute_f1())
