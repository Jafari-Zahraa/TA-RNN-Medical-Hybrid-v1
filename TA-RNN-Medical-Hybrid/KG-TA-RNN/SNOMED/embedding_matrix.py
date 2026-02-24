# 1_build_embedding_matrix.py
import json
import numpy as np
import pickle
import pandas as pd


def build_embedding_matrix():
    """
    ساخت ماتریس برای لایه Embedding مدل
    """
    # 1. بارگذاری merged embeddings
    with open('embeddings/snomed_merged_embeddings.json', 'r') as f:
        merged_embeddings = json.load(f)  # SNOMED → Vector

    # 2. بارگذاری mapping کامل (از mapper_final_full.py)
    mapping_df = pd.read_csv('icdToSnomedMapping/icd_to_snomed_mapping_complete.csv')
    # ستون‌ها: icd, numeric_code, snomed_codes, mapped

    # 3. بارگذاری types (کدهای عددی ICD)
    with open('mimic/mimic_output.types', 'rb') as f:
        types = pickle.load(f)  # {'D_250.10': 0, 'D_401.9': 1, ...}

    # 4. ایجاد ماتریس خالی
    embedding_dim = len(next(iter(merged_embeddings.values())))  # مثلاً 192
    num_codes = len(types)

    # +1 برای padding (سطر ۰ همیشه صفر)
    embedding_matrix = np.zeros((num_codes + 1, embedding_dim))

    # 5. پر کردن ماتریس
    for _, row in mapping_df.iterrows():
        icd_code = row['icd']  # مثلاً 'D_250.10'
        snomed_codes = row['snomed_codes']  # مثلاً '169132001,146350008'
        numeric_id = row['numeric_code']  # مثلاً 0

        if row['mapped'] and snomed_codes:
            # استخراج لیست SNOMEDها
            snomed_list = snomed_codes.split(',')

            # جمع‌آوری embeddings مربوطه
            embeddings = []
            for snomed in snomed_list:
                if snomed in merged_embeddings:
                    embeddings.append(merged_embeddings[snomed])

            if embeddings:
                # میانگین گرفتن
                avg_embedding = np.mean(embeddings, axis=0)
                embedding_matrix[numeric_id + 1] = avg_embedding  # +1 چون سطر ۰ برای padding
            else:
                # اگر icdToSnomedMapping پیدا نشد، تصادفی
                embedding_matrix[numeric_id + 1] = np.random.normal(0, 0.1, embedding_dim)

    # 6. ذخیره
    np.save('dataScience/snomed_embedding_matrix.npy', embedding_matrix)
    print(f"✅ Embedding Matrix ساخته شد: {embedding_matrix.shape}")

    return embedding_matrix

if __name__ == "__main__":
    build_embedding_matrix();