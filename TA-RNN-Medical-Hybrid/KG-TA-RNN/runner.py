"""
disease_analysis.py
Main disease analysis file - clean English version
"""

import json
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from DiseaseInterpreter import DiseaseInterpreter
from keras.models import load_model
from TimeEmbeddingLayer import Time_embedding_layer
from AttentionLayer import AttentionLayer
from LossFunction import binary_cross_entropy

# Import modular components
from Html_Reporter import HTMLReporter

from GlobalVariablesMIMIC import (
    lon_data_last,
    time_data_last,
    demo_data_last,rid_last
)


# =========================
# Helper Functions
# =========================

def prepare_patient_input(lon_data, time_data, demo_data, patient_idx, max_visits, demo_dim):
    """Prepare patient input for interpretation"""
    # 1. Longitudinal ICD data
    lon = np.asarray(lon_data[0][patient_idx], dtype=np.int32)
    T, F = lon.shape

    if T > max_visits:
        lon = lon[:max_visits, :]
    else:
        lon = np.pad(lon, ((0, max_visits - T), (0, 0)), constant_values=0)

    lon = lon.reshape(1, max_visits, F)

    # 2. Time data
    time = np.asarray(time_data[patient_idx], dtype=np.float32)
    if time.ndim == 2:
        time = time[:, 0]

    if time.shape[0] > max_visits:
        time = time[:max_visits]
    else:
        time = np.pad(time, (0, max_visits - time.shape[0]), constant_values=0.0)

    time = time.reshape(1, max_visits)

    # 3. Demographic data
    demo = np.asarray(demo_data[0][patient_idx], dtype=np.float32)
    if demo.shape[0] > demo_dim:
        demo = demo[:demo_dim]
    else:
        demo = np.pad(demo, (0, demo_dim - demo.shape[0]), constant_values=0.0)

    demo = demo.reshape(1, demo_dim)

    return [lon, time, demo]



class PatientManager:
    """Manage patient folders - one folder per patient"""

    def __init__(self, base_dir="patient_results"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def get_patient_dir(self, patient_id, overwrite=False):
        """Get patient directory path - removes existing if overwrite=True"""
        patient_dir = os.path.join(self.base_dir, patient_id)

        if overwrite and os.path.exists(patient_dir):
            print(f"⚠️  Removing existing directory for {patient_id}")
            shutil.rmtree(patient_dir)

        os.makedirs(patient_dir, exist_ok=True)
        return patient_dir

    def list_patients(self):
        """List all analyzed patients"""
        if not os.path.exists(self.base_dir):
            return []

        patients = []
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                patients.append(item)

        return sorted(patients)

    def get_patient_info(self, patient_id):
        """Get patient analysis information"""
        patient_dir = os.path.join(self.base_dir, patient_id)
        if not os.path.exists(patient_dir):
            return None

        info = {
            'patient_id': patient_id,
            'directory': patient_dir,
            'created': datetime.fromtimestamp(os.path.getctime(patient_dir)).strftime('%Y-%m-%d %H:%M:%S'),
            'files': []
        }

        for root, dirs, files in os.walk(patient_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, patient_dir)
                info['files'].append({
                    'name': rel_path,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                })

        return info


class ResultSaver:
    """Save interpretation results to patient folder"""

    def __init__(self, patient_id, patient_manager):
        self.patient_id = patient_id
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.patient_manager = patient_manager
        self.patient_dir = patient_manager.get_patient_dir(patient_id, overwrite=True)

        # Create subdirectories
        self._create_subdirectories()

    def _create_subdirectories(self):
        """Create internal folder structure"""
        subdirs = ['json', 'reports', 'plots', 'data', 'html', 'archive']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.patient_dir, subdir), exist_ok=True)

    def archive_previous_results(self):
        """Archive previous analysis results"""
        archive_dir = os.path.join(self.patient_dir, 'archive', self.timestamp)
        os.makedirs(archive_dir, exist_ok=True)

        # Move old files to archive (except archive itself)
        for item in os.listdir(self.patient_dir):
            if item != 'archive':
                item_path = os.path.join(self.patient_dir, item)
                if os.path.isfile(item_path):
                    shutil.move(item_path, os.path.join(archive_dir, item))

        print(f"📦 Archived previous results to: {archive_dir}")

    def convert_for_json(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_for_json(item) for item in obj]
        return obj

    def save_interpretation_json(self, interpretation):
        """Save complete interpretation as JSON"""
        json_path = os.path.join(self.patient_dir, 'json', 'interpretation.json')

        interpretation_json = self.convert_for_json(interpretation)
        interpretation_json['metadata'] = {
            'export_timestamp': self.timestamp,
            'patient_id': self.patient_id,
            'analysis_version': '2.0',
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(interpretation_json, f, indent=2, ensure_ascii=False)

        print(f"   ✅ JSON saved: {json_path}")
        return json_path

    def save_clinical_insights(self, clinical_insights, mortality_risk, risk_category):
        """Save clinical insights separately"""
        insights_path = os.path.join(self.patient_dir, 'json', 'clinical_insights.json')

        insights_data = {
            'patient_id': self.patient_id,
            'timestamp': self.timestamp,
            'mortality_risk': float(mortality_risk),
            'risk_category': risk_category,
            'insights': clinical_insights,
            'priority_summary': {
                'critical': len([i for i in clinical_insights if i['priority'] == 'critical']),
                'high': len([i for i in clinical_insights if i['priority'] == 'high']),
                'medium': len([i for i in clinical_insights if i['priority'] == 'medium'])
            }
        }

        with open(insights_path, 'w', encoding='utf-8') as f:
            json.dump(insights_data, f, indent=2, ensure_ascii=False)

        print(f"   ✅ Clinical insights saved: {insights_path}")
        return insights_path

    def save_text_report(self, interpretation):
        """Save text clinical report"""
        report_path = os.path.join(self.patient_dir, 'reports', 'clinical_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"PATIENT CLINICAL REPORT - {self.patient_id}\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Patient ID: {interpretation['patient_id']}\n")
            f.write(f"Analysis Date: {self.timestamp}\n")
            f.write(f"Number of Visits: {interpretation['num_visits']}\n")
            f.write(f"Mortality Risk: {interpretation['mortality_risk']:.2%}\n")
            f.write(f"Risk Category: {interpretation['risk_category'].upper()}\n\n")

            # Top diseases
            f.write("-" * 60 + "\n")
            f.write("DISEASE PROFILE (Top 5)\n")
            f.write("-" * 60 + "\n")
            for i, (disease, info) in enumerate(list(interpretation['disease_profile'].items())[:5]):
                f.write(f"{i + 1}. {disease}\n")
                f.write(f"   - Score: {info['score']:.3f}\n")
                f.write(f"   - Severity: {info['severity']}\n")
                f.write(f"   - Visit Frequency: {info['frequency']}\n")
                f.write(f"   - Chronic: {'Yes' if info['is_chronic'] else 'No'}\n\n")

            # Clinical insights
            f.write("-" * 60 + "\n")
            f.write("CLINICAL INSIGHTS\n")
            f.write("-" * 60 + "\n")
            if interpretation['clinical_insights']:
                for insight in interpretation['clinical_insights']:
                    f.write(f"● [{insight['priority'].upper()}] {insight['message']}\n")
                    f.write(f"  → Suggested Action: {insight['suggested_action']}\n\n")
            else:
                f.write("No clinical insights generated.\n\n")

        print(f"   ✅ Text report saved: {report_path}")
        return report_path

    def save_plots(self, interpretation):
        """Save visualization plots"""
        plots_dir = os.path.join(self.patient_dir, 'plots')

        # Disease progression plot
        prog_plot_path = os.path.join(plots_dir, 'disease_progression.png')

        if 'disease_progression' in interpretation and interpretation['disease_progression']:
            plt.figure(figsize=(12, 6))

            for i, (disease, prog) in enumerate(list(interpretation['disease_progression'].items())[:5]):
                values = prog['values']
                plt.plot(range(1, len(values) + 1), values,
                         marker='o', linewidth=2, label=f"{disease[:30]} ({prog['trend']})")

            plt.title(f"Disease Progression - {self.patient_id}", fontsize=14)
            plt.xlabel("Visit Number")
            plt.ylabel("Importance Score")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(prog_plot_path, dpi=200)
            plt.close()

            print(f"   ✅ Progression plot saved: {prog_plot_path}")

        return plots_dir

    def save_input_data(self, patient_inputs):
        """Save raw input data"""
        if patient_inputs is not None:
            data_path = os.path.join(self.patient_dir, 'data', 'patient_inputs.npz')
            np.savez(data_path,
                     visit_codes=patient_inputs[0],
                     time_data=patient_inputs[1],
                     demo_data=patient_inputs[2])

            print(f"   ✅ Input data saved: {data_path}")
            return data_path
        return None

    def save_summary(self):
        """Save summary file"""
        summary_path = os.path.join(self.patient_dir, 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"""PATIENT ANALYSIS SUMMARY
=======================
Patient ID: {self.patient_id}
Analysis Timestamp: {self.timestamp}
Directory: {self.patient_dir}

Generated Files:
- interpretation.json (Complete interpretation data)
- clinical_insights.json (Clinical insights)
- clinical_report.txt (Text report)
- disease_progression.png (Progression plot)
- interpretation_summary.png (Main visualization)
- patient_inputs.npz (Input data)

To view the report, open: html/clinical_dashboard.html
""")

        print(f"   ✅ Summary saved: {summary_path}")
        return summary_path

    def create_html_report(self, interpretation):
        """Create HTML report"""
        html_reporter = HTMLReporter(self.patient_dir)
        html_path = html_reporter.create_dashboard(interpretation, self.timestamp)

        print(f"   ✅ HTML report saved: {html_path}")
        return html_path

    def get_patient_dir(self):
        """Get patient directory path"""
        return self.patient_dir


# =========================
# Main Analysis Functions
# =========================

def load_model_and_interpreter(model_path):
    """Load model and create interpreter"""
    try:
        model = load_model(
            model_path,
            custom_objects={
                "Time_embedding_layer": Time_embedding_layer,
                "AttentionLayer": AttentionLayer,
                "binary_cross_entropy": binary_cross_entropy
            },
            compile=False
        )

        interpreter = DiseaseInterpreter(model=model)
        return model, interpreter
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None


def analyze_patient(patient_idx, model, interpreter):
    """Analyze specific patient"""
    # Prepare input
    max_visits = model.input_shape[0][1]
    demo_dim = model.input_shape[2][1] if len(model.input_shape) > 2 else 10

    patient_inputs = prepare_patient_input(
        lon_data_last,
        time_data_last,
        demo_data_last,
        patient_idx=patient_idx,
        max_visits=max_visits,
        demo_dim=demo_dim
    )

    # Interpret patient
    interpretation = interpreter.interpret_patient(
        patient_inputs,
        patient_id=f"PATIENT_{patient_idx}"
    )

    return interpretation, patient_inputs


def save_all_results(interpretation, interpreter, patient_inputs=None, patient_manager=None):
    """Save all analysis results"""
    if patient_manager is None:
        patient_manager = PatientManager()

    # Create saver for patient
    saver = ResultSaver(interpretation['patient_id'], patient_manager)

    # Archive previous results
    saver.archive_previous_results()

    # Save all components
    saver.save_interpretation_json(interpretation)
    saver.save_clinical_insights(
        interpretation['clinical_insights'],
        interpretation['mortality_risk'],
        interpretation['risk_category']
    )
    saver.save_text_report(interpretation)
    saver.save_plots(interpretation)
    saver.save_input_data(patient_inputs)

    # Create HTML report
    saver.create_html_report(interpretation)

    # Save summary
    saver.save_summary()

    # Create main visualization
    main_plot_path = os.path.join(saver.get_patient_dir(), 'plots', 'interpretation_summary.png')
    interpreter.visualize_interpretation(interpretation, save_path=main_plot_path)

    return saver.get_patient_dir()


def analyze_single_patient(patient_idx=2, model_path="saved_models/TA_RNN_iter_1.h5", overwrite=True):
    """Analyze single patient"""
    print("=" * 60)
    print(f"ANALYZING PATIENT MIMIC_{patient_idx}")
    print("=" * 60)

    try:
        # 1. Patient management
        patient_manager = PatientManager()

        # 2. Load model
        print("1. Loading model...")
        model, interpreter = load_model_and_interpreter(model_path)
        if model is None or interpreter is None:
            return {'success': False, 'error': 'Failed to load model'}

        # 3. Analyze patient
        print(f"2. Analyzing patient MIMIC_{patient_idx}...")
        interpretation, patient_inputs = analyze_patient(patient_idx, model, interpreter)

        # 4. Save results
        print("3. Saving results...")
        patient_dir = save_all_results(
            interpretation,
            interpreter,
            patient_inputs,
            patient_manager
        )

        # 5. Display results
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"✅ Patient: {interpretation['patient_id']}")
        print(f"📊 Mortality Risk: {interpretation['mortality_risk']:.2%} ({interpretation['risk_category'].upper()})")
        print(f"🏥 Diseases Identified: {len(interpretation['disease_profile'])}")
        print(f"📈 Clinical Insights: {len(interpretation['clinical_insights'])}")
        print(f"\n📁 Patient Directory: {patient_dir}")
        print(f"🌐 HTML Report: {patient_dir}/html/clinical_dashboard.html")
        print("=" * 60)

        return {
            'success': True,
            'patient_id': interpretation['patient_id'],
            'mortality_risk': float(interpretation['mortality_risk']),
            'risk_category': interpretation['risk_category'],
            'patient_dir': patient_dir,
            'html_report': f"{patient_dir}/html/clinical_dashboard.html"
        }

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def analyze_multiple_patients(patient_indices, model_path="saved_models/TA_RNN_iter_1.h5"):
    """Analyze multiple patients"""
    print("=" * 60)
    print(f"ANALYZING {len(patient_indices)} PATIENTS")
    print("=" * 60)

    results = []
    patient_manager = PatientManager()

    for idx, patient_idx in enumerate(patient_indices, 1):
        print(f"\n[{idx}/{len(patient_indices)}] Processing patient MIMIC_{patient_idx}")

        try:
            # Load model
            model, interpreter = load_model_and_interpreter(model_path)
            if model is None:
                print(f"   ❌ Failed to load model for patient {patient_idx}")
                continue

            # Analyze patient
            interpretation, patient_inputs = analyze_patient(patient_idx, model, interpreter)

            # Save results
            patient_dir = save_all_results(
                interpretation,
                interpreter,
                patient_inputs,
                patient_manager
            )

            results.append({
                'patient_id': interpretation['patient_id'],
                'mortality_risk': float(interpretation['mortality_risk']),
                'risk_category': interpretation['risk_category'],
                'disease_count': len(interpretation['disease_profile']),
                'patient_dir': patient_dir,
                'success': True
            })

            print(f"   ✅ Completed: {interpretation['patient_id']} - Risk: {interpretation['mortality_risk']:.2%}")

        except Exception as e:
            print(f"   ❌ Error processing patient {patient_idx}: {e}")
            results.append({
                'patient_id': f"PATIENT_{patient_idx}",
                'success': False,
                'error': str(e)
            })

    # Display summary
    print("\n" + "=" * 60)
    print("BATCH ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total patients processed: {len(patient_indices)}")
    print(f"Successful analyses: {len([r for r in results if r.get('success')])}")
    print(f"Failed analyses: {len([r for r in results if not r.get('success')])}")

    # Risk summary
    successful_results = [r for r in results if r.get('success')]
    if successful_results:
        print(f"\nRisk Distribution:")
        high_risk = len([r for r in successful_results if r.get('risk_category') == 'high'])
        low_risk = len([r for r in successful_results if r.get('risk_category') == 'low'])
        print(f"  High risk: {high_risk} patients")
        print(f"  Low risk: {low_risk} patients")

    print(f"\n📁 All results saved in: patient_results/")
    print("=" * 60)

    return results


def list_analyzed_patients():
    """List all analyzed patients"""
    patient_manager = PatientManager()
    patients = patient_manager.list_patients()

    print("=" * 60)
    print("ANALYZED PATIENTS")
    print("=" * 60)

    if not patients:
        print("No patients analyzed yet.")
        return []

    for i, patient_id in enumerate(patients, 1):
        info = patient_manager.get_patient_info(patient_id)
        if info:
            print(f"{i}. {patient_id}")
            print(f"   📅 Created: {info['created']}")
            print(f"   📁 Directory: {info['directory']}")
            print(f"   📊 Files: {len(info['files'])}")
            print()

    print(f"Total: {len(patients)} patients")
    print("=" * 60)

    return patients


# =========================
# Main Execution
# =========================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Disease Interpretation Analysis (Interactive Mode)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="saved_models/TA_RNN_iter_1.h5",
        help="Path to trained model"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all previously analyzed patients"
    )

    args = parser.parse_args()

    # -------------------------------------------------
    # List analyzed patients
    # -------------------------------------------------
    if args.list:
        list_analyzed_patients()
        sys.exit(0)

    # -------------------------------------------------
    # Interactive patient input
    # -------------------------------------------------
    print("=" * 60)
    print("DISEASE INTERPRETATION SYSTEM")
    print("=" * 60)

    try:
        patient_idx = int(
            input("Please enter patient index (integer): ").strip()
        )
    except ValueError:
        print("❌ Invalid input. Patient index must be an integer.")
        sys.exit(1)

    print(f"\nSelected patient index: {patient_idx}")
    print("=" * 60)

    # -------------------------------------------------
    # Run analysis
    # -------------------------------------------------
    result = analyze_single_patient(
        patient_idx=patient_idx,
        model_path=args.model
    )

    # -------------------------------------------------
    # Final status
    # -------------------------------------------------
    if result.get("success"):
        print("\n✅ Analysis completed successfully.")
        print(f"Patient ID     : {result['patient_id']}")
        print(f"Mortality Risk : {result['mortality_risk']:.2%}")
        print(f"Risk Category  : {result['risk_category'].upper()}")
        print(f"Results Folder: {result['patient_dir']}")
        print(f"HTML Report   : {result['html_report']}")
    else:
        print("\n❌ Analysis failed.")
        print(f"Error: {result.get('error')}")
