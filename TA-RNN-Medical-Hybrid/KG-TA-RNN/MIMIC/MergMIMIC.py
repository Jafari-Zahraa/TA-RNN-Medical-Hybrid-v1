import pandas as pd
import pickle


def create_demographic_data(patients_file, output_pids_file, output_demographic_file):
    # Read patient IDs
    with open(output_pids_file, 'rb') as f:
        patient_ids = pickle.load(f)

    # Read patients data
    patients_df = pd.read_csv(patients_file, compression='gzip')

    # Filter and rename
    demographic_df = patients_df[patients_df['subject_id'].isin(patient_ids)].copy()
    demographic_df = demographic_df.rename(columns={'subject_id': 'RID'})

    # Remove mortality column if exists
    if 'dod' in demographic_df.columns:
        demographic_df = demographic_df.drop('dod', axis=1)

    # Save to CSV
    demographic_df.to_csv(output_demographic_file, index=False)
    print(f"Demographic data saved: {output_demographic_file}")

    return demographic_df


def check_demographic_dataset(df):
    if df.empty:
        print('Dataset is empty.')
        return -1

    if df.isnull().sum().any():
        print('Dataset has missing values.')
        return -1

    if 'RID' not in df.columns:
        print('Dataset does not have RID feature')
        return -1

    if not pd.api.types.is_numeric_dtype(df['RID']):
        print('RID should be numeric.')
        return -1

    if df.shape[1] <= 1:
        print('Dataset does not have enough features')
        return -1

    return 1


def clean_longitudinal_data(df):
    """
    Clean longitudinal data by removing rows with all empty ICD codes
    """
    print("Cleaning longitudinal data...")

    # پیدا کردن ستون‌های ICD
    icd_columns = [col for col in df.columns if col.startswith('icd_code')]

    # فقط ردیف‌هایی که حداقل یک کد ICD دارند نگه دار
    before_rows = len(df)

    # ایجاد ماسک: ردیف‌هایی که حداقل یک کد ICD غیرخالی دارند
    has_icd_mask = df[icd_columns].apply(lambda x: any(pd.notna(val) and val != '' and val != 0 for val in x), axis=1)
    df_clean = df[has_icd_mask].copy()

    after_rows = len(df_clean)
    removed_rows = before_rows - after_rows

    print(f"Before cleaning: {before_rows} rows")
    print(f"After cleaning: {after_rows} rows")
    print(f"Removed rows: {removed_rows} rows")

    # اطمینان از numeric بودن کدهای ICD
    for col in icd_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)

    return df_clean


def create_longitudinal_data(output_file_base, output_csv_file):
    """
    Create longitudinal data with one row per visit
    Merge: pids, dates, seqs_3digit, morts
    Calculate VISCODE as time difference between visits
    Expand ICD codes into multiple columns
    """

    # Read all pickle files
    with open(output_file_base + '.pids', 'rb') as f:
        pids = pickle.load(f)

    with open(output_file_base + '.dates', 'rb') as f:
        dates = pickle.load(f)

    with open(output_file_base + '.3digitICD9.seqs', 'rb') as f:
        seqs_3digit = pickle.load(f)

    with open(output_file_base + '.morts', 'rb') as f:
        morts = pickle.load(f)

    # Prepare data for CSV
    longitudinal_data = []

    for i, pid in enumerate(pids):
        patient_visits = dates[i]
        patient_diagnoses = seqs_3digit[i]
        mortality = morts[i]

        for j, visit_date in enumerate(patient_visits):
            visit_diagnoses = patient_diagnoses[j]

            # Calculate VISCODE (time difference from first visit in months)
            if j == 0:
                viscode = 0
            else:
                time_diff = visit_date - patient_visits[0]
                viscode = round(time_diff.days / 30.0)

            # Create row data
            row_data = {
                'RID': pid,
                'VISCODE': viscode,
                'mortality': mortality
            }

            # Add ICD codes as separate columns
            for k, code in enumerate(visit_diagnoses):
                row_data[f'icd_code{k + 1}'] = code

            longitudinal_data.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(longitudinal_data)

    # Fill NaN values with 0 for ICD code columns
    icd_columns = [col for col in df.columns if col.startswith('icd_code')]
    df[icd_columns] = df[icd_columns].fillna(0)

    # 🔥 CLEAN THE DATA - اضافه کردن این خط
    df = clean_longitudinal_data(df)

    # Save to CSV
    df.to_csv(output_csv_file, index=False)
    print(f"Longitudinal data saved: {output_csv_file}")
    print(f"Total visits: {len(df)}")
    print(f"Total patients: {len(pids)}")
    print(f"ICD code columns: {len(icd_columns)}")

    # Display sample
    print("\nSample data:")
    print(df.head())

    return df


# Usage
output_file_base = 'CleanData/mimic_output'
output_csv_file = 'CleanData/longitudinal_data.csv'

longitudinal_df = create_longitudinal_data(output_file_base, output_csv_file)

# Display sample
print("\nSample data:")
print(longitudinal_df.columns)
print(longitudinal_df.head(10))

# Usage
patients_file = 'RawData/patients.csv.gz'
output_pids_file = 'CleanData/mimic_output.pids'
output_demographic_file = 'CleanData/demographic_data.csv'

demographic_data = create_demographic_data(patients_file, output_pids_file, output_demographic_file)
check_result = check_demographic_dataset(demographic_data)