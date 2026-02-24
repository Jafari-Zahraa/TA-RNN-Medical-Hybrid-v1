# mapper_final_full_complete.py
import pickle, json, re, logging, os, traceback
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# -------------------------------
# Logger
# -------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------------------------------
# Mapper Class - COMPLETE VERSION
# -------------------------------
class ICDSNOMEDMapper:

    def __init__(self, data_dir=".", output_dir="icdToSnomedMapping"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.icd_to_snomed = {}
        self.numeric_to_snomed = {}
        self.normalization_cache = {}

    # -------------------------------
    # 1️⃣ Load MIMIC ICD
    # -------------------------------
    def load_mimic_types(self):
        types_path = self.data_dir / "mimic/mimic_output.types"
        if not types_path.exists():
            raise FileNotFoundError(f"Types file not found: {types_path}")
        with open(types_path, "rb") as f:
            types = pickle.load(f)
        types_reverse = {v: k for k, v in types.items()}
        return types, types_reverse

    def load_mimic_sequences(self):
        seqs_path = self.data_dir / "mimic/mimic_output.seqs"
        pids_path = self.data_dir / "mimic/mimic_output.pids"
        with open(seqs_path, "rb") as f:
            seqs = pickle.load(f)
        with open(pids_path, "rb") as f:
            pids = pickle.load(f)
        return seqs, pids

    # -------------------------------
    # 2️⃣ ICD Normalization
    # -------------------------------
    def normalize_icd(self, icd):
        if not icd or pd.isna(icd):
            return ""
        if icd in self.normalization_cache:
            return self.normalization_cache[icd]
        icd_clean = str(icd).strip().upper()
        icd_clean = re.sub(r"^(ICD9|ICD10|D_)", "", icd_clean)
        icd_clean = re.sub(r"[^\w\.]", "", icd_clean)
        icd_norm = icd_clean.replace(".", "")
        self.normalization_cache[icd] = icd_norm
        return icd_norm

    # -------------------------------
    # 3️⃣ Load SNOMED maps
    # -------------------------------
    def load_snomed_maps(self, snomed_dir="Terminology"):
        snomed_dir = Path(snomed_dir)
        files = list(snomed_dir.glob("der2_*MapSnapshot_*.txt"))
        if not files:
            raise FileNotFoundError("No SNOMED mapping files found")
        self.icd_to_snomed = defaultdict(set)
        for file in tqdm(files, desc="Loading SNOMED maps"):
            df = pd.read_csv(file, sep="\t", dtype=str, low_memory=False)
            if "active" in df.columns:
                df = df[df["active"] == "1"]
            for _, r in df.iterrows():
                icd_candidates = []
                for col in ["mapTarget", "mapSource", "sourceId"]:
                    if col in r:
                        icd_candidates.append(str(r[col]))
                snomed_candidates = []
                for col in ["referencedComponentId", "destinationId", "target"]:
                    if col in r:
                        snomed_candidates.append(str(r[col]))
                for icd in icd_candidates:
                    icd_norm = self.normalize_icd(icd)
                    for sn in snomed_candidates:
                        if icd_norm and sn:
                            self.icd_to_snomed[icd_norm].add(sn)
        # Convert sets to lists
        self.icd_to_snomed = {k: list(v) for k, v in self.icd_to_snomed.items()}
        logger.info(f"✅ Loaded {len(self.icd_to_snomed):,} ICD codes from SNOMED maps")

    # -------------------------------
    # 4️⃣ Mapping with fallback
    # -------------------------------
    def map_icd(self, icd):
        icd_norm = self.normalize_icd(icd)
        if not icd_norm:
            return []
        # Exact match
        if icd_norm in self.icd_to_snomed:
            return self.icd_to_snomed[icd_norm]
        # Remove last digits (prefix fallback)
        for i in range(len(icd_norm) - 1, 2, -1):
            prefix = icd_norm[:i]
            if prefix in self.icd_to_snomed:
                return self.icd_to_snomed[prefix]
        return []

    # -------------------------------
    # 5️⃣ Run mapping for all codes
    # -------------------------------
    def run_mapping(self, types_reverse):
        results = {}
        for num_code, icd in tqdm(types_reverse.items(), desc="Mapping ICD codes"):
            snomed = self.map_icd(icd)
            results[icd] = {
                "numeric_code": num_code,
                "original_icd": icd,
                "snomed_codes": snomed,
                "mapped": bool(snomed)
            }
            if snomed:
                self.numeric_to_snomed[num_code] = snomed[0]
        return results

    # -------------------------------
    # 6️⃣ Save outputs
    # -------------------------------
    def save_results(self, results):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        df_rows = []
        for icd, info in results.items():
            df_rows.append({
                "icd": icd,
                "numeric_code": info["numeric_code"],
                "snomed_codes": ",".join(info["snomed_codes"]),
                "mapped": info["mapped"]
            })
        df = pd.DataFrame(df_rows)
        df.to_csv(self.output_dir / "icd_to_snomed_mapping_complete.csv", index=False)
        df[df["mapped"]].to_csv(self.output_dir / "icd_to_snomed_mapped_only.csv", index=False)
        df[~df["mapped"]].to_csv(self.output_dir / "unmapped_icd_codes.csv", index=False)
        with open(self.output_dir / "numeric_to_snomed_quick_map.json", "w", encoding="utf-8") as f:
            json.dump(self.numeric_to_snomed, f, indent=2)
        logger.info(f"✅ All outputs saved.")

    # -------------------------------
    # 7️⃣ Create snomed_descriptions.json
    # -------------------------------
    def create_snomed_descriptions_json(self,
                                        desc_file="Terminology/sct2_Description_Snapshot-en_INT_20251201.txt",
                                        output_file="Terminology/snomed_descriptions.json"):
        """
        ساخت فایل JSON از تمام توضیحات SNOMED
        خروجی: {"snomed_code": "description", ...}
        """
        print("\n" + "=" * 60)
        print("📝 CREATING snomed_descriptions.json")
        print("=" * 60)

        descriptions = {}

        try:
            # 1. پیدا کردن فایل descriptions
            desc_path = Path(desc_file)
            if not desc_path.exists():
                print(f"   ⚠️ Description file not found: {desc_path}")
                # جستجوی فایل‌های دیگر
                term_dir = Path("Terminology")
                alt_files = list(term_dir.rglob("*Description*.txt"))
                if alt_files:
                    desc_path = alt_files[0]
                    print(f"   🔍 Found alternative: {desc_path}")
                else:
                    print("   ❌ No SNOMED description files found")
                    return None

            # 2. خواندن فایل
            print(f"   📖 Reading: {desc_path}")

            # تلاش با جداکننده‌های مختلف
            separators = ['\t', ',', ';', '|']
            df = None

            for sep in separators:
                try:
                    df = pd.read_csv(desc_path, sep=sep, dtype=str, low_memory=False,
                                     on_bad_lines='skip', encoding='utf-8')
                    print(f"   ✅ Success with separator: '{sep}'")
                    break
                except:
                    continue

            if df is None:
                print("   ❌ Could not read file with any separator")
                return None

            print(f"   📊 File shape: {df.shape}")

            # 3. بررسی ستون‌ها
            available_cols = df.columns.tolist()
            print(f"   📋 Available columns ({len(available_cols)}): {available_cols}")

            # پیدا کردن ستون conceptId
            concept_col_candidates = ['conceptId', 'referencedComponentId', 'conceptid',
                                      'concept_id', 'concept', 'id']
            concept_col = None
            for col in concept_col_candidates:
                if col in available_cols:
                    concept_col = col
                    break

            # پیدا کردن ستون term
            term_col_candidates = ['term', 'term_string', 'description', 'termdescription',
                                   'descriptionterm', 'name', 'label']
            term_col = None
            for col in term_col_candidates:
                if col in available_cols:
                    term_col = col
                    break

            if not concept_col or not term_col:
                print(f"   ⚠️ Required columns not found.")
                print(f"   Concept candidates: {concept_col_candidates}")
                print(f"   Term candidates: {term_col_candidates}")
                print(f"   Using first two columns as fallback...")
                concept_col = available_cols[0]
                term_col = available_cols[1] if len(available_cols) > 1 else available_cols[0]

            print(f"   🔧 Using columns: '{concept_col}' → '{term_col}'")

            # 4. فیلتر active records
            if 'active' in available_cols:
                initial_count = len(df)
                df = df[df['active'] == '1']
                print(f"   ✅ Filtered to active records: {len(df)} (from {initial_count})")

            # 5. استخراج descriptions
            records_processed = 0
            duplicates = 0

            for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting descriptions"):
                snomed_code = row.get(concept_col)
                term_text = row.get(term_col)

                if pd.notna(snomed_code) and pd.notna(term_text):
                    snomed_code = str(snomed_code).strip()
                    term_text = str(term_text).strip()

                    if snomed_code and term_text:
                        # فقط اگر کد معتبر SNOMED است
                        if snomed_code.isdigit() and len(snomed_code) >= 6:
                            if snomed_code not in descriptions:
                                descriptions[snomed_code] = term_text
                                records_processed += 1
                            else:
                                duplicates += 1

            # 6. ذخیره به JSON
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True, parents=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(descriptions, f, indent=2, ensure_ascii=False)

            print(f"\n✅ SUCCESS: Created {output_path}")
            print(f"   Total unique SNOMED descriptions: {len(descriptions):,}")
            print(f"   Records processed: {records_processed:,}")
            print(f"   Duplicates skipped: {duplicates:,}")

            # نمایش نمونه
            if descriptions:
                print(f"\n   Sample entries:")
                sample_items = list(descriptions.items())[:5]
                for code, desc in sample_items:
                    print(f"     {code}: {desc[:80]}...")

            return descriptions

        except Exception as e:
            print(f"❌ Error creating snomed_descriptions.json: {e}")
            traceback.print_exc()
            return None

    # -------------------------------
    # 8️⃣ Create English mapping with descriptions
    # -------------------------------
    def create_english_mapping_with_descriptions(self, results, descriptions_dict):
        """
        ساخت mapping انگلیسی با توضیحات کامل
        """
        print("\n" + "=" * 60)
        print("🔗 CREATING ENGLISH MAPPING WITH DESCRIPTIONS")
        print("=" * 60)

        icd_to_english = {}
        snomed_to_english = {}

        # اگر descriptions نداریم، از fallback استفاده می‌کنیم
        if not descriptions_dict:
            print("   ⚠️ No SNOMED descriptions available, using fallback...")
            descriptions_dict = {}

        unmapped_count = 0
        descriptions_used = 0

        for icd, info in tqdm(results.items(), desc="Creating English mapping"):
            english_name = None
            snomed_codes = info['snomed_codes']

            # روش ۱: از SNOMED descriptions
            if snomed_codes and descriptions_dict:
                for snomed in snomed_codes:
                    if snomed in descriptions_dict:
                        english_name = descriptions_dict[snomed]
                        snomed_to_english[snomed] = english_name
                        descriptions_used += 1
                        break

            # روش ۲: fallback based on ICD code
            if not english_name:
                icd_clean = icd.replace('D_', '')

                # تشخیص بر اساس کدهای رایج
                if icd_clean.startswith('250'):
                    if '250.0' in icd or '250.00' in icd or '250.01' in icd:
                        english_name = "Type 1 Diabetes"
                    elif '250.1' in icd or '250.10' in icd or '250.11' in icd:
                        english_name = "Type 2 Diabetes"
                    else:
                        english_name = "Diabetes Mellitus"

                elif icd_clean.startswith('401'):
                    english_name = "Hypertension"
                elif icd_clean.startswith('428'):
                    english_name = "Heart Failure"
                elif icd_clean.startswith('414'):
                    english_name = "Coronary Artery Disease"
                elif icd_clean.startswith('496'):
                    english_name = "Chronic Obstructive Pulmonary Disease"
                elif icd_clean.startswith('585'):
                    english_name = "Chronic Kidney Disease"
                elif icd_clean.startswith('272'):
                    english_name = "Disorder of Lipoprotein Metabolism"
                elif icd_clean.startswith('493'):
                    english_name = "Asthma"
                elif icd_clean.startswith('427'):
                    english_name = "Cardiac Arrhythmia"
                elif icd_clean.startswith('571'):
                    english_name = "Liver Disease"
                elif icd_clean.startswith('331'):
                    english_name = "Dementia"
                elif icd_clean.startswith('434'):
                    english_name = "Cerebral Infarction"
                elif icd_clean.startswith('041'):
                    english_name = "Bacterial Infection"
                elif icd_clean.startswith('038'):
                    english_name = "Sepsis"
                elif icd_clean.startswith('V'):
                    english_name = f"Medical Service/Status: {icd_clean}"
                elif icd_clean.startswith('E'):
                    english_name = f"External Cause: {icd_clean}"
                else:
                    english_name = f"Disease (ICD-9: {icd_clean})"
                    unmapped_count += 1

            icd_to_english[icd] = english_name

        # 3. ذخیره فایل‌ها
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # الف) فایل اصلی: ICD → English Name
        english_df = pd.DataFrame([
            {
                'icd': icd,
                'english_name': name,
                'numeric_code': results[icd]['numeric_code'],
                'mapped': results[icd]['mapped'],
                'snomed_codes': ','.join(results[icd]['snomed_codes']) if results[icd]['snomed_codes'] else '',
                'has_snomed_description': 'Yes' if any(
                    s in descriptions_dict for s in results[icd]['snomed_codes']) else 'No'
            }
            for icd, name in icd_to_english.items()
        ])

        english_file = self.output_dir / "icd_to_english_mapping.csv"
        english_df.to_csv(english_file, index=False, encoding='utf-8')

        # ب) فایل JSON برای استفاده راحت‌تر
        json_file = self.output_dir / "icd_to_english.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(icd_to_english, f, indent=2, ensure_ascii=False)

        # ج) فایل SNOMED → English (اگر داده داشتیم)
        if snomed_to_english:
            snomed_df = pd.DataFrame([
                {'snomed_code': code, 'english_name': name}
                for code, name in snomed_to_english.items()
            ])
            snomed_file = self.output_dir / "snomed_to_english_mapping.csv"
            snomed_df.to_csv(snomed_file, index=False, encoding='utf-8')

        print(f"✅ Created English mapping for {len(icd_to_english)} ICD codes")
        print(f"   SNOMED descriptions used: {descriptions_used:,}")
        print(f"   Fallback naming used: {unmapped_count:,}")
        print(f"   Output files:")
        print(f"     - {english_file}")
        print(f"     - {json_file}")
        if snomed_to_english:
            print(f"     - {snomed_file}")

        return icd_to_english, snomed_to_english

    # -------------------------------
    # 9️⃣ Create complete mapping with descriptions
    # -------------------------------
    def create_complete_mapping_with_descriptions(self, results, descriptions_dict):
        """
        ساخت mapping کامل با توضیحات انگلیسی
        """
        print("\n" + "=" * 60)
        print("📊 CREATING COMPLETE MAPPING WITH DESCRIPTIONS")
        print("=" * 60)

        complete_mapping = []
        descriptions_available = 0

        for icd, info in tqdm(results.items(), desc="Building complete mapping"):
            icd_info = {
                'icd': icd,
                'numeric_code': info['numeric_code'],
                'snomed_codes': info['snomed_codes'],
                'mapped': info['mapped'],
                'snomed_descriptions': [],
                'primary_description': '',
                'description_source': 'none'
            }

            # اضافه کردن توضیحات SNOMED
            if info['snomed_codes'] and descriptions_dict:
                for snomed in info['snomed_codes']:
                    if snomed in descriptions_dict:
                        icd_info['snomed_descriptions'].append({
                            'snomed_code': snomed,
                            'description': descriptions_dict[snomed]
                        })
                        descriptions_available += 1

                # انتخاب primary description
                if icd_info['snomed_descriptions']:
                    icd_info['primary_description'] = icd_info['snomed_descriptions'][0]['description']
                    icd_info['description_source'] = 'snomed'

            complete_mapping.append(icd_info)

        # ذخیره به JSON
        json_file = self.output_dir / "complete_icd_mapping_with_descriptions.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(complete_mapping, f, indent=2, ensure_ascii=False)

        # همچنین به صورت CSV ذخیره کن
        df_complete = pd.DataFrame([
            {
                'icd': item['icd'],
                'numeric_code': item['numeric_code'],
                'snomed_codes': ','.join(item['snomed_codes']),
                'primary_description': item['primary_description'],
                'description_source': item['description_source'],
                'mapped': item['mapped']
            }
            for item in complete_mapping
        ])

        csv_file = self.output_dir / "complete_icd_mapping_with_descriptions.csv"
        df_complete.to_csv(csv_file, index=False, encoding='utf-8')

        print(f"✅ Created complete mapping:")
        print(f"   Total ICD codes: {len(complete_mapping):,}")
        print(f"   SNOMED descriptions available: {descriptions_available:,}")
        print(f"   Output files:")
        print(f"     - {json_file}")
        print(f"     - {csv_file}")

        return complete_mapping


# -------------------------------
# Main Execution - COMPLETE VERSION
# -------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 ICD TO SNOMED MAPPING PIPELINE - COMPLETE VERSION")
    print("=" * 70)

    # ایجاد mapper
    mapper = ICDSNOMEDMapper(data_dir=".", output_dir="icdToSnomedMapping")

    # 1. بارگذاری types
    print("\n1️⃣ LOADING MIMIC ICD TYPES")
    print("-" * 40)
    types, types_reverse = mapper.load_mimic_types()
    print(f"   ✅ Loaded {len(types):,} ICD codes")

    # 2. بارگذاری mapsهای SNOMED
    print("\n2️⃣ LOADING SNOMED MAPPING FILES")
    print("-" * 40)
    mapper.load_snomed_maps("Terminology")

    # 3. انجام mapping
    print("\n3️⃣ MAPPING ICD TO SNOMED")
    print("-" * 40)
    results = mapper.run_mapping(types_reverse)

    # 4. ذخیره نتایج اصلی
    print("\n4️⃣ SAVING BASIC RESULTS")
    print("-" * 40)
    mapper.save_results(results)

    # 5. ساخت snomed_descriptions.json
    print("\n5️⃣ CREATING SNOMED DESCRIPTIONS JSON")
    print("-" * 40)
    descriptions = mapper.create_snomed_descriptions_json(
        desc_file="Terminology/sct2_Description_Snapshot-en_INT_20251201.txt",
        output_file="Terminology/snomed_descriptions.json"
    )

    # 6. ساخت mapping انگلیسی
    print("\n6️⃣ CREATING ENGLISH MAPPING")
    print("-" * 40)
    icd_to_english, snomed_to_english = mapper.create_english_mapping_with_descriptions(results, descriptions)

    # 7. ساخت mapping کامل
    print("\n7️⃣ CREATING COMPLETE MAPPING")
    print("-" * 40)
    complete_mapping = mapper.create_complete_mapping_with_descriptions(results, descriptions)

    # 8. گزارش نهایی
    print("\n" + "=" * 70)
    print("🎉 FINAL REPORT")
    print("=" * 70)

    mapped_count = sum(1 for r in results.values() if r["mapped"])
    coverage = mapped_count / len(results) * 100

    print(f"📊 ICD CODES:")
    print(f"   Total: {len(results):,}")
    print(f"   Mapped to SNOMED: {mapped_count:,} ({coverage:.1f}%)")
    print(f"   Unmapped: {len(results) - mapped_count:,}")

    print(f"\n📝 DESCRIPTIONS:")
    print(f"   SNOMED descriptions loaded: {len(descriptions) if descriptions else 0:,}")

    if icd_to_english:
        english_with_desc = sum(1 for icd in icd_to_english if 'Disease (ICD-9:' not in icd_to_english[icd])
        print(f"   ICD codes with English names: {english_with_desc:,}/{len(icd_to_english):,}")

    print(f"\n📁 OUTPUT FILES CREATED:")
    print(f"   📂 {mapper.output_dir}/")
    print(f"     ├── icd_to_snomed_mapping_complete.csv")
    print(f"     ├── icd_to_snomed_mapped_only.csv")
    print(f"     ├── unmapped_icd_codes.csv")
    print(f"     ├── numeric_to_snomed_quick_map.json")
    print(f"     ├── icd_to_english_mapping.csv")
    print(f"     ├── icd_to_english.json")
    print(f"     ├── complete_icd_mapping_with_descriptions.json")
    print(f"     └── complete_icd_mapping_with_descriptions.csv")

    if descriptions:
        print(f"   📂 Terminology/")
        print(f"     └── snomed_descriptions.json")

    print("\n✅ PIPELINE COMPLETE!")
    print("=" * 70)