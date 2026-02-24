import numpy as np
import pandas as pd
import pickle

from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.model_selection import train_test_split

# =====================================================
# Validation
# =====================================================
def check_longitudinal_dataset(df):
    if df.empty:
        raise ValueError("Longitudinal dataset is empty")
    for c in ['RID', 'VISCODE', 'mortality']:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")
    if not is_numeric_dtype(df['RID']):
        raise ValueError("RID must be numeric")
    return True


def check_demographic_dataset(df):
    if df.empty:
        raise ValueError("Demographic dataset is empty")
    if 'RID' not in df.columns:
        raise ValueError("RID missing in demographic data")
    if not is_numeric_dtype(df['RID']):
        raise ValueError("RID must be numeric")
    return True


# =====================================================
# VISCODE → regular timeline (6 months)
# =====================================================
def visit_code_preparation(df):
    df = df.sort_values(['RID', 'VISCODE']).reset_index(drop=True)
    df['VISCODE'] = df.groupby('RID').cumcount() * 6
    return df


# =====================================================
# Filter patients by minimum visits
# =====================================================
def filter_patients_by_min_visits(df, min_visits):
    visit_counts = df.groupby('RID').size()
    valid_rids = visit_counts[visit_counts >= min_visits].index
    return df[df['RID'].isin(valid_rids)]


# =====================================================
# Group patients by number of visits
# =====================================================
def group_patients_according_number_of_visits(df):
    visits_dic = {}
    for rid in df['RID'].unique():
        temp = df[df['RID'] == rid].copy()
        n = len(temp)
        visits_dic.setdefault(n, []).append(temp)
    for n in visits_dic:
        visits_dic[n] = pd.concat(visits_dic[n], ignore_index=True)
    return dict(sorted(visits_dic.items()))


# =====================================================
# Transpose longitudinal data (NO mortality in features)
# =====================================================
def transpose_longitudinal_data(group_dic, features):
    out = {}
    for k, df in group_dic.items():
        rows = []
        cols = (
            ['RID']
            + [f"{f}_{i}" for i in range(k) for f in features]
            + [f"mortality_{i}" for i in range(k)]
        )

        for rid, g in df.groupby('RID'):
            g = g.sort_values('VISCODE').reset_index(drop=True)
            row = [rid]

            for i in range(k):
                for f in features:
                    val = g.loc[i, f]
                    row.append(0.0 if pd.isna(val) else val)

            for i in range(k):
                row.append(g.loc[i, 'mortality'])

            rows.append(row)

        out[k] = pd.DataFrame(rows, columns=cols)

    return out


# =====================================================
# Demographic one-hot encoding
# =====================================================
def demographic_one_hot_encoding(df):
    df = df.copy()
    cat_cols = [c for c in df.columns if is_string_dtype(df[c]) and c != 'RID']
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c]) and c != 'RID']
    encoded = pd.get_dummies(df[cat_cols], prefix=cat_cols)
    final_df = pd.concat([df[['RID']], df[num_cols], encoded], axis=1)
    final_df = final_df.drop_duplicates('RID').set_index('RID').sort_index()
    return final_df


# =====================================================
# REAL Sliding Window Dataset Creation
# =====================================================
def create_dataset_sliding(dfs, ts, fts, demographic_df):
    X, y, demo = [], [], []

    for df in dfs:
        feat_cols = [c for c in df.columns if '_' in c and not c.startswith('mortality')]
        label_cols = [c for c in df.columns if c.startswith('mortality_')]

        base_features = sorted(set(c.rsplit('_', 1)[0] for c in feat_cols))
        max_visits = max(int(c.rsplit('_', 1)[1]) for c in label_cols) + 1

        for i in range(len(df)):
            rid = df.loc[i, 'RID']

            for start in range(0, max_visits - ts - fts + 1):

                # -------- X --------
                seq = []
                for t in range(start, start + ts):
                    step = [df.loc[i, f"{f}_{t}"] for f in base_features]
                    seq.append(step)
                X.append(seq)

                # -------- y --------
                future = [
                    df.loc[i, f"mortality_{t}"]
                    for t in range(start + ts, start + ts + fts)
                ]
                y.append(future)

                # -------- demographic --------
                if rid in demographic_df.index:
                    demo.append(demographic_df.loc[rid].values)
                else:
                    demo.append(np.zeros(demographic_df.shape[1]))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, fts, 1)

    return X, y, demo


# =====================================================
# Elapsed Time (normalized)
# =====================================================
def elapsed_time_from_viscode(num_samples, ts):
    max_time = (ts - 1) * 6
    ela = []
    for _ in range(num_samples):
        if max_time == 0:
            ela.append([[0.0]] * ts)
        else:
            ela.append([[ (i * 6) / max_time ] for i in range(ts)])
    return np.array(ela, dtype=np.float32)


# =====================================================
# LAST Window Dataset (Patient-level for Interpretation)
# =====================================================
def create_last_window_dataset(df, features, ts, fts, demographic_df):
    """
    df          : DataFrame اصلی (بعد از فیلتر، قبل از transpose) با ستون VISCODE
    features    : لیست feature ها (بدون RID, VISCODE, mortality)
    ts          : تعداد ویزیت های تاریخچه (input sequence length)
    fts         : تعداد ویزیت های آینده (برای sanity check)
    demographic_df : DataFrame دمگرافیک (index=RID)
    """
    X_last, y_last, demo_last, rid_last = [], [], [], []

    for rid, g in df.groupby('RID'):
        g = g.sort_values('VISCODE').reset_index(drop=True)
        n_visits = len(g)

        if n_visits < ts:
            continue  # بیمار پنجره کافی نداره

        # آخرین ts ویزیت
        last_window = g.iloc[-ts:]

        # Features
        X_row = last_window[features].fillna(0.0).values  # shape: (ts, num_features)
        X_last.append(X_row)

        # Labels (sanity check)
        y_future = []
        for i in range(fts):
            idx = -fts + i
            if abs(idx) <= n_visits:
                y_future.append(last_window.iloc[idx]['mortality'])
            else:
                y_future.append(0.0)
        y_last.append(y_future)

        # RID
        rid_last.append(rid)

        # Demographic
        if rid in demographic_df.index:
            demo_last.append(demographic_df.loc[rid].values)
        else:
            demo_last.append(np.zeros(demographic_df.shape[1]))

    X_last = np.array(X_last, dtype=np.float32)
    y_last = np.array(y_last, dtype=np.float32).reshape(-1, fts, 1)
    demo_last = np.array(demo_last, dtype=np.float32)

    return X_last, y_last, demo_last, rid_last


# =====================================================
# Main
# =====================================================
def pkl_files_creator():

    longitudinal_df = pd.read_csv('CleanData/longitudinal_data.csv')
    demographic_df = pd.read_csv('CleanData/demographic_data.csv')

    check_longitudinal_dataset(longitudinal_df)
    check_demographic_dataset(demographic_df)

    longitudinal_df = visit_code_preparation(longitudinal_df)

    # ❗ mortality کاملاً حذف شده از features
    features = [
        c for c in longitudinal_df.columns
        if c not in ['RID', 'VISCODE', 'mortality']
    ]

    ts = int(input("Enter number of history visits (ts): "))
    fts = int(input("Enter number of future visits (fts): "))

    # ---------------- Split by RID ----------------
    rids = longitudinal_df['RID'].unique()
    train_rids, test_rids = train_test_split(rids, test_size=0.3, random_state=42)

    train_df = longitudinal_df[longitudinal_df['RID'].isin(train_rids)]
    test_df  = longitudinal_df[longitudinal_df['RID'].isin(test_rids)]

    # ---------------- Filter ----------------
    train_df = filter_patients_by_min_visits(train_df, ts + fts)
    test_df  = filter_patients_by_min_visits(test_df, ts + fts)

    # ---------------- Group + Transpose ----------------
    train_groups = group_patients_according_number_of_visits(train_df)
    test_groups  = group_patients_according_number_of_visits(test_df)

    train_data = transpose_longitudinal_data(train_groups, features)
    test_data  = transpose_longitudinal_data(test_groups, features)

    # ---------------- Demographic ----------------
    demographic_df = demographic_one_hot_encoding(demographic_df)

    demo_array = demographic_df.values.astype(np.float32)
    train_idx = demographic_df.index.isin(train_df['RID'].unique())
    numeric_cols = [0, 1]
    mean = demo_array[train_idx][:, numeric_cols].mean(axis=0)
    std  = demo_array[train_idx][:, numeric_cols].std(axis=0)
    demo_array[:, numeric_cols] = (demo_array[:, numeric_cols] - mean) / (std + 1e-8)
    demographic_df = pd.DataFrame(demo_array, index=demographic_df.index, columns=demographic_df.columns)

    # ---------------- Sliding Window Dataset ----------------
    X_train, y_train, demo_train = create_dataset_sliding(list(train_data.values()), ts, fts, demographic_df)
    X_test, y_test, demo_test = create_dataset_sliding(list(test_data.values()), ts, fts, demographic_df)

    ela_train = elapsed_time_from_viscode(X_train.shape[0], ts)
    ela_test  = elapsed_time_from_viscode(X_test.shape[0], ts)


    X_last, y_last, demo_last, rid_last = create_last_window_dataset(
        test_df,  # <--- توجه: DataFrame اصلی، قبل از transpose
        features,
        ts,
        fts,
        demographic_df
    )

    # Elapsed Time
    ela_last = elapsed_time_from_viscode(len(X_last), ts)

    # ---------------- Save PKL ----------------
    with open('DataScience/longitudinal_data_train.pkl','wb') as f:
        pickle.dump([X_train], f)
    with open('DataScience/label_train.pkl','wb') as f:
        pickle.dump([y_train], f)
    with open('DataScience/demographic_data_train.pkl','wb') as f:
        pickle.dump([demo_train], f)
    with open('DataScience/elapsed_data_train.pkl','wb') as f:
        pickle.dump(ela_train, f)

    with open('DataScience/longitudinal_data_test.pkl','wb') as f:
        pickle.dump([X_test], f)
    with open('DataScience/label_test.pkl','wb') as f:
        pickle.dump([y_test], f)
    with open('DataScience/demographic_data_test.pkl','wb') as f:
        pickle.dump([demo_test], f)
    with open('DataScience/elapsed_data_test.pkl','wb') as f:
        pickle.dump(ela_test, f)

    with open('DataScience/lon_data_last.pkl', 'wb') as f:
        pickle.dump([X_last], f)
    with open('DataScience/label_last.pkl', 'wb') as f:
        pickle.dump([y_last], f)
    with open('DataScience/demo_data_last.pkl', 'wb') as f:
        pickle.dump([demo_last], f)
    with open('DataScience/rid_last.pkl', 'wb') as f:
        pickle.dump(rid_last, f)
    with open('DataScience/elapsed_last.pkl', 'wb') as f:
        pickle.dump(ela_last, f)

    print("✅ FINAL sliding-window dataset successfully created")


# =====================================================
# Run
# =====================================================
if __name__ == "__main__":
    pkl_files_creator()
