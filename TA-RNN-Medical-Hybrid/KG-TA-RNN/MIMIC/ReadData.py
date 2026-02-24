import pickle
import os
from datetime import datetime
import pandas as pd
import numpy as np
import os

df = pd.read_csv("RawData/DIAGNOSES_ICD.csv.gz")
print("Columns DIAGNOSES:")
print(df.columns.tolist())
print("\nSample rows:")
print(df.head())

df = pd.read_csv("RawData/patients.csv.gz")
print("Columns patients:")
print(df.columns.tolist())
print("\nSample rows:")
print(df.head())

df = pd.read_csv("RawData/admissions.csv.gz")
print("Columns admissions:")
print(df.columns.tolist())
print(df.head())

df = pd.read_csv("CleanData/demographic_data.csv")
print("demographic_data:")
print(df.columns.tolist())
print(df.head())

df = pd.read_csv("CleanData/longitudinal_data.csv")
print("longitudinal_data:")
print(df.columns.tolist())
print(df.head())
file_name = 'DataScience/longitudinal_data_train.pkl'
lon_data_train = pd.read_pickle(file_name)
longitudinal_train_data = lon_data_train[0]
print("=== longitudinal_train_data ===")
print("Type:", type(longitudinal_train_data))
print("Shape:", longitudinal_train_data.shape)
print("First 5 samples:\n", longitudinal_train_data[:5])

# ************************************************************************

# Labels of traing data
file_name = 'DataScience/label_train.pkl'
label_train = pd.read_pickle(file_name)
train_label = label_train[0]
print("=== train_label ===")
print("Type:", type(train_label))
print("Shape:", train_label.shape)
print("First 5 samples:\n", train_label[:5])

# ************************************************************************

# Demographic training data
file_name = 'DataScience/demographic_data_train.pkl'
dem_data_train = pd.read_pickle(file_name)
demographic_train_data = np.array(dem_data_train[0])
print("=== demographic_train_data ===")
print("Type:", type(demographic_train_data))
print("Shape:", demographic_train_data.shape)
print("First 5 samples:\n", demographic_train_data[:5])

# ************************************************************************

# elapsed time training data
file_name = 'DataScience/elapsed_data_train.pkl'
time_train = pd.read_pickle(file_name)
train_time = time_train
train_time = np.reshape(train_time, (train_time.shape[0], train_time.shape[1] * train_time.shape[2]))
print("=== train_time ===")
print("Type:", type(train_time))
print("Shape:", train_time.shape)
print("First 5 samples:\n", train_time[:5])

# ************************************************************************

# Longitudinal test data
file_name = 'DataScience/longitudinal_data_test.pkl'
lon_data_test = pd.read_pickle(file_name)
longitudinal_test_data = lon_data_test[0]
print("=== longitudinal_test_data ===")
print("Type:", type(longitudinal_test_data))
if hasattr(longitudinal_test_data, 'shape'):
    print("Shape:", longitudinal_test_data.shape)
print("First 5 samples:\n", longitudinal_test_data[:5])
print("\n" + "="*60 + "\n")
# ************************************************************************

# Labels of test data
file_name = 'DataScience/label_test.pkl'
label_test = pd.read_pickle(file_name)
test_label = label_test[0]

print("===test_label ===")
print("Type:", type(test_label))
if hasattr(test_label, 'shape'):
    print("Shape:", test_label.shape)
print("First 5 samples:\n", test_label[:5])
print("\n" + "="*60 + "\n")

# ************************************************************************

# Demographic test data
file_name = 'DataScience/demographic_data_test.pkl'
dem_data_test = pd.read_pickle(file_name)
demographic_test_data = np.array(dem_data_test[0])
print("=== demographic_test_data ===")
print("Type:", type(demographic_test_data))
print("Shape:", demographic_test_data.shape)
print("First 5 samples:\n", demographic_test_data[:5])
print("\n" + "="*60 + "\n")

# ************************************************************************

# elapsed time test data
file_name = 'DataScience/elapsed_data_test.pkl'
time_test = pd.read_pickle(file_name)
test_time = time_test
test_time = np.reshape(test_time, (test_time.shape[0], test_time.shape[1] * test_time.shape[2]))
print("=== test_time ===")
print("Type:", type(test_time))
print("Shape:", test_time.shape)
print("First 5 samples:\n", test_time[:5])


def display_samples(data):
    print("=" * 70)
    print("نمونه‌های داده‌های MIMIC-III")
    print("=" * 70)

    # 1. نمایش samples از pids
    print("\n1. فایل *.pids (شناسه بیماران)")
    print(f"   تعداد کل: {len(data['pids'])} بیمار")
    print(f"   ۵ نمونه اول: {data['pids'][:5]}")

    # 2. نمایش samples از morts
    print("\n2. فایل *.morts (وضعیت فوت)")
    print(f"   تعداد کل: {len(data['morts'])} رکورد")
    mort_count = sum(data['morts'])
    print(f"   تعداد فوت‌شدگان: {mort_count} ({mort_count / len(data['morts']) * 100:.1f}%)")
    print(f"   ۵ نمونه اول: {data['morts'][:5]}")

    # 3. نمایش samples از dates
    print("\n3. فایل *.dates (تاریخ ویزیت‌ها)")
    print(f"   تعداد کل: {len(data['dates'])} بیمار")
    total_visits = sum(len(visits) for visits in data['dates'])
    print(f"   تعداد کل ویزیت‌ها: {total_visits}")
    print(f"   میانگین ویزیت per بیمار: {total_visits / len(data['dates']):.1f}")
    print("   ۵ نمونه اول:")
    for i, patient_dates in enumerate(data['dates'][:5]):
        print(f"      بیمار {i + 1}: {len(patient_dates)} ویزیت")
        for j, date in enumerate(patient_dates[:3]):  # حداکثر ۳ تاریخ نمایش
            print(f"        ویزیت {j + 1}: {date}")

    # 4. نمایش samples از seqs
    print("\n4. فایل *.seqs (توالی تشخیص‌ها - کدهای عددی)")
    print(f"   تعداد کل: {len(data['seqs'])} بیمار")
    total_diagnoses = sum(len(visit) for patient in data['seqs'] for visit in patient)
    print(f"   تعداد کل تشخیص‌ها: {total_diagnoses}")
    print("   ۵ نمونه اول:")
    for i, patient_seqs in enumerate(data['seqs'][:5]):
        print(f"      بیمار {i + 1}: {len(patient_seqs)} ویزیت")
        for j, visit in enumerate(patient_seqs[:3]):  # حداکثر ۳ ویزیت نمایش
            print(f"        ویزیت {j + 1}: {len(visit)} تشخیص → کدها: {visit}")

    # 5. نمایش samples از types
    print("\n5. فایل *.types (دیکشنری کدهای تشخیصی)")
    print(f"   تعداد کل کدهای تشخیصی: {len(data['types'])}")
    print("   ۱۰ کد اول:")
    items = list(data['types'].items())[:10]
    for code, num in items:
        print(f"      {num}: {code}")

    # 6. نمایش samples از seqs_3digit
    print("\n6. فایل *.3digitICD9.seqs (کدهای ۳رقمی)")
    print(f"   تعداد کل: {len(data['seqs_3digit'])} بیمار")
    total_3digit_diagnoses = sum(len(visit) for patient in data['seqs_3digit'] for visit in patient)
    print(f"   تعداد کل تشخیص‌های ۳رقمی: {total_3digit_diagnoses}")
    print("   ۵ نمونه اول:")
    for i, patient_seqs in enumerate(data['seqs_3digit'][:5]):
        print(f"      بیمار {i + 1}: {len(patient_seqs)} ویزیت")
        for j, visit in enumerate(patient_seqs[:2]):  # حداکثر ۲ ویزیت نمایش
            print(f"        ویزیت {j + 1}: {len(visit)} تشخیص → کدها: {visit}")

    # 7. نمایش samples از types_3digit
    print("\n7. فایل *.3digitICD9.types (دیکشنری کدهای ۳رقمی)")
    print(f"   تعداد کل کدهای ۳رقمی: {len(data['types_3digit'])}")
    print("   ۱۰ کد اول:")
    items_3digit = list(data['types_3digit'].items())[:10]
    for code, num in items_3digit:
        print(f"      {num}: {code}")

    # 8. اطلاعات آماری کلی
    print("\n" + "=" * 70)
    print("خلاصه آماری")
    print("=" * 70)

    # توزیع تعداد ویزیت‌ها
    visit_counts = [len(visits) for visits in data['dates']]
    print(f"تعداد بیماران: {len(data['pids'])}")
    print(f"تعداد کل ویزیت‌ها: {sum(visit_counts)}")
    print(f"میانگین ویزیت per بیمار: {sum(visit_counts) / len(visit_counts):.2f}")
    print(f"حداکثر ویزیت یک بیمار: {max(visit_counts)}")
    print(f"حداقل ویزیت یک بیمار: {min(visit_counts)}")

    # توزیع تعداد تشخیص‌ها per ویزیت
    diagnosis_counts = [len(visit) for patient in data['seqs'] for visit in patient]
    if diagnosis_counts:
        print(f"میانگین تشخیص per ویزیت: {sum(diagnosis_counts) / len(diagnosis_counts):.2f}")
        print(f"حداکثر تشخیص در یک ویزیت: {max(diagnosis_counts)}")
        print(f"حداقل تشخیص در یک ویزیت: {min(diagnosis_counts)}")


def load_all_data(base_path):
    """بارگذاری همه فایل‌های داده"""
    data = {}
    files = {
        'pids': '.pids',
        'morts': '.morts',
        'dates': '.dates',
        'seqs': '.seqs',
        'types': '.types',
        'seqs_3digit': '.3digitICD9.seqs',
        'types_3digit': '.3digitICD9.types'
    }

    print("در حال بارگذاری فایل‌ها...")
    for key, ext in files.items():
        file_path = base_path + ext
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    data[key] = pickle.load(f)
                print(f"✓ {key} - اندازه: {len(data[key])}")
            except Exception as e:
                print(f"✗ خطا در خواندن {key}: {e}")
                data[key] = None
        else:
            print(f"✗ فایل وجود ندارد: {file_path}")
            data[key] = None

    return data


# اجرای اصلی
if __name__ == "__main__":
    # مسیر فایل‌های شما - اینجا باید اصلاح شود
    base_path = r'D:\app\PythonProject\TA-RNN\dataset\MIMIC\CleanData\mimic_output'

    print("شروع بارگذاری داده‌های MIMIC-III...")
    data = load_all_data(base_path)

    if data['pids'] is not None:
        print("\n" + "=" * 70)
        print("بارگذاری کامل شد! نمایش نمونه‌ها...")
        print("=" * 70)
        display_samples(data)
    else:
        print("\nخطا: فایل‌ها بارگذاری نشدند. لطفاً مسیر را بررسی کنید.")