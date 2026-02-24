# disease_interpreter_final.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import pickle
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path


class DiseaseInterpreter:
    """
    DiseaseInterpreter (Corrected & Stable)

    - Visit-level attention analysis (alpha)
    - ICD-based disease attribution (NO fake embedding-dim mapping)
    - Disease profile
    - Disease progression
    - Clinical insights
    - Text report
    - Visualization
    """

    def __init__(self, model_path=None, model=None):
        print("🧠 Initializing Disease Interpreter...")

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = tf.keras.models.load_model(model_path, compile=False)
        else:
            raise ValueError("Either model or model_path must be provided")

        print(f"   Model loaded: {self.model.name}")

        self._load_mappings()
        self._load_embedding_matrix()
        self.interpreter_model = self._build_interpreter_model()

        print("✅ Disease Interpreter initialized successfully!")

    # ------------------------------------------------------------------
    # Loading resources
    # ------------------------------------------------------------------

    def _load_mappings(self):
        print("🗺️ Loading ICD mappings...")

        icd_english_path = Path("SNOMED/icdToSnomedMapping/icd_to_english.json")
        self.icd_to_english = (
            json.load(open(icd_english_path, "r", encoding="utf-8"))
            if icd_english_path.exists()
            else {}
        )

        types_path = Path("MIMIC/CleanData/mimic_output.types")
        if not types_path.exists():
            raise FileNotFoundError("mimic_output.types not found")

        with open(types_path, "rb") as f:
            self.types = pickle.load(f)

        self.types_reverse = {v: k for k, v in self.types.items()}

    def _load_embedding_matrix(self):
        emb_path = Path("SNOMED/DataScience/snomed_embedding_matrix.npy")
        if not emb_path.exists():
            raise FileNotFoundError("snomed_embedding_matrix.npy not found")

        self.embedding_matrix = np.load(emb_path)
        self.embedding_dim = self.embedding_matrix.shape[1]

    # ------------------------------------------------------------------
    # Interpreter model
    # ------------------------------------------------------------------

    def _build_interpreter_model(self):
        attention_layer = None
        for layer in self.model.layers:
            if "attention" in layer.name.lower():
                attention_layer = layer
                break

        if attention_layer is None:
            raise ValueError("No attention layer found in model")

        outputs = {
            "alpha": attention_layer.output[0],
            "beta": attention_layer.output[1],
            "context": attention_layer.output[2],
            "prediction": self.model.output,
        }

        return Model(inputs=self.model.inputs, outputs=outputs)

    # ------------------------------------------------------------------
    # Main interpretation
    # ------------------------------------------------------------------

    def interpret_patient(self, patient_inputs, patient_id=None):
        outputs = self.interpreter_model.predict(patient_inputs, verbose=0)

        alpha = outputs["alpha"]      # (1, T, 1)
        beta = outputs["beta"]        # (1, T, D)
        prediction = outputs["prediction"]

        visit_codes = patient_inputs[0][0]  # (T, F)

        interpretation = {
            "patient_id": patient_id or f"patient_{np.random.randint(10000,99999)}",
            "mortality_risk": float(prediction[0][0]),
            "risk_category": self._categorize_mortality_risk(float(prediction[0][0])),
            "num_visits": visit_codes.shape[0],
            "important_visits": [],
            "disease_profile": {},
            "disease_progression": {},
            "clinical_insights": [],
            "attention_analysis": {
                "alpha_mean": float(alpha.mean()),
                "alpha_std": float(alpha.std()),
                "beta_mean": float(beta.mean()),
                "beta_std": float(beta.std()),
            },
        }

        interpretation = self._analyze_visits(
            interpretation, alpha, beta, visit_codes
        )
        interpretation = self._build_disease_profile(
            interpretation, alpha, beta, visit_codes
        )
        interpretation = self._analyze_progression(
            interpretation, alpha, beta, visit_codes
        )
        interpretation = self._generate_clinical_insights(interpretation)

        return interpretation

    # ------------------------------------------------------------------
    # Visit-level analysis
    # ------------------------------------------------------------------

    def _analyze_visits(self, interpretation, alpha, beta, visit_codes):
        alpha_vals = alpha[0, :, 0]
        T, F = visit_codes.shape

        for t in range(T):
            if np.all(visit_codes[t] == 0):
                continue

            visit_info = {
                "visit_index": t,
                "visit_importance": float(alpha_vals[t]),
                "top_diseases": [],
            }

            for icd_id in visit_codes[t]:
                if icd_id == 0 or icd_id >= self.embedding_matrix.shape[0]:
                    continue

                emb = self.embedding_matrix[int(icd_id)]
                emb_norm = np.linalg.norm(emb)

                importance = float(alpha_vals[t] * emb_norm)

                if importance <= 0:
                    continue

                icd_code = self.types_reverse.get(int(icd_id), f"ICD_{icd_id}")
                name = self.icd_to_english.get(icd_code, icd_code)

                visit_info["top_diseases"].append({
                    "icd": icd_code,
                    "disease_name": name,
                    "importance": importance,
                })

            if visit_info["top_diseases"]:
                visit_info["top_diseases"].sort(
                    key=lambda x: x["importance"], reverse=True
                )
                interpretation["important_visits"].append(visit_info)

        return interpretation

    # ------------------------------------------------------------------
    # Disease profile
    # ------------------------------------------------------------------

    def _build_disease_profile(self, interpretation, alpha, beta, visit_codes):
        scores = defaultdict(float)
        visits = defaultdict(set)

        T, F = visit_codes.shape

        for t in range(T):
            alpha_v = float(alpha[0, t, 0])

            for icd_id in visit_codes[t]:
                if icd_id == 0 or icd_id >= self.embedding_matrix.shape[0]:
                    continue

                emb = self.embedding_matrix[int(icd_id)]
                emb_norm = np.linalg.norm(emb)

                s = alpha_v * emb_norm
                if s <= 0:
                    continue

                icd_code = self.types_reverse.get(int(icd_id), f"ICD_{icd_id}")
                name = self.icd_to_english.get(icd_code, icd_code)

                scores[name] += s
                visits[name].add(t)

        if scores:
            max_score = max(scores.values())
            for d, s in scores.items():
                norm = s / max_score if max_score > 0 else 0
                interpretation["disease_profile"][d] = {
                    "score": norm,
                    "raw_score": s,
                    "severity": self._categorize_severity(norm),
                    "frequency": len(visits[d]),
                    "is_chronic": len(visits[d]) >= max(2, T - 1),
                }

        interpretation["disease_profile"] = dict(
            sorted(
                interpretation["disease_profile"].items(),
                key=lambda x: x[1]["score"],
                reverse=True,
            )
        )

        return interpretation

    # ------------------------------------------------------------------
    # Disease progression
    # ------------------------------------------------------------------

    def _analyze_progression(self, interpretation, alpha, beta, visit_codes):
        T, F = visit_codes.shape

        for disease in list(interpretation["disease_profile"].keys())[:10]:
            values = []

            for t in range(T):
                beta_v = beta[0, t]
                alpha_v = float(alpha[0, t, 0])
                score = 0.0

                for icd_id in visit_codes[t]:
                    if icd_id == 0 or icd_id >= self.embedding_matrix.shape[0]:
                        continue

                    icd_code = self.types_reverse.get(int(icd_id), f"ICD_{icd_id}")
                    name = self.icd_to_english.get(icd_code, icd_code)

                    if name == disease:
                        emb = self.embedding_matrix[int(icd_id)]
                        score += alpha_v * np.linalg.norm(emb)

                values.append(score)

            if any(v > 0 for v in values):
                interpretation["disease_progression"][disease] = {
                    "values": values,
                    "trend": self._calculate_trend(values),
                    "peak_visit": int(np.argmax(values)),
                    "peak_value": float(np.max(values)),
                }

        return interpretation

    # ------------------------------------------------------------------
    # Clinical insights
    # ------------------------------------------------------------------

    def _generate_clinical_insights(self, interpretation):
        insights = []

        risk = interpretation["mortality_risk"]
        category = interpretation["risk_category"]

        if category == "critical":
            insights.append({
                "priority": "critical",
                "message": "Very high mortality risk detected",
                "suggested_action": "Immediate clinical review and ICU-level attention",
            })

        elif category == "high":
            insights.append({
                "priority": "high",
                "message": "High mortality risk detected",
                "suggested_action": "Close monitoring and senior clinician review",
            })

        elif category == "moderate":
            insights.append({
                "priority": "moderate",
                "message": "Moderate mortality risk detected",
                "suggested_action": "Assess risk factors and adjust care plan",
            })

        else:  # low
            insights.append({
                "priority": "low",
                "message": "Low mortality risk",
                "suggested_action": "Routine clinical care",
            })

        # --- existing severe disease logic (UNCHANGED) ---
        severe = [
            d for d, i in interpretation["disease_profile"].items()
            if i["severity"] == "severe"
        ]
        if severe:
            insights.append({
                "priority": "high",
                "message": f"Severe conditions: {', '.join(severe[:3])}",
                "suggested_action": "Specialist consultation",
            })

        interpretation["clinical_insights"] = insights
        return interpretation

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def generate_report(self, interpretation, save_path=None):
        lines = []
        lines.append(f"Patient ID: {interpretation['patient_id']}")
        lines.append(f"Mortality risk: {interpretation['mortality_risk']:.2%}")
        lines.append(f"Number of visits: {interpretation['num_visits']}")

        if interpretation["disease_profile"]:
            lines.append("\nTop Diseases:")
            for d, info in list(interpretation["disease_profile"].items())[:5]:
                lines.append(
                    f"- {d}: score={info['score']:.2f}, severity={info['severity']}"
                )

        text = "\n".join(lines)

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text)

        return text

    def _categorize_severity(self, score):
        if score > 0.7:
            return "severe"
        elif score > 0.4:
            return "moderate"
        return "mild"

    def _categorize_mortality_risk(self, risk):
        """
        4-level mortality risk categorization (Option A)
        """
        if risk < 0.20:
            return "low"
        elif risk < 0.40:
            return "moderate"
        elif risk < 0.70:
            return "high"
        else:
            return "critical"

    def _calculate_trend(self, values):
        if len(values) < 2:
            return "stable"
        x = np.arange(len(values))
        y = np.array(values)
        if np.count_nonzero(y) < 2:
            return "stable"
        slope = np.polyfit(x[y > 0], y[y > 0], 1)[0]
        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        return "stable"

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_interpretation(self, interpretation, save_path=None):
        """
        Visualization with colored Mortality Risk bar:
        1) Mortality Risk
        2) Disease Profile
        3) Important Visits
        """

        import matplotlib.patches as mpatches

        num_visits = interpretation["num_visits"]
        x_visits = np.arange(1, num_visits + 1)

        plt.figure(figsize=(12, 10))

        # ==================================
        # 1. Mortality Risk (Colored)
        # ==================================
        plt.subplot(3, 1, 1)

        mortality = interpretation["mortality_risk"] * 100

        # Determine color
        if mortality < 20:
            color = "green"
        elif mortality < 40:
            color = "yellow"
        elif mortality < 80:
            color = "orange"
        else:
            color = "red"

        plt.bar(["Mortality Risk"], [mortality], color=color)
        plt.ylim(0, 100)
        plt.ylabel("Percent (%)", fontsize=9)
        plt.title("Predicted Mortality Risk", fontsize=10)
        plt.text(
            0,
            mortality + 2,
            f"{mortality:.1f}%",
            ha="center",
            fontsize=9
        )

        # Legend
        legend_patches = [
            mpatches.Patch(color="green", label="<20%"),
            mpatches.Patch(color="yellow", label="20-40%"),
            mpatches.Patch(color="orange", label="40-80%"),
            mpatches.Patch(color="red", label=">80%"),
        ]
        plt.legend(handles=legend_patches, fontsize=8)

        # ==================================
        # 2. Disease Profile
        # ==================================
        plt.subplot(3, 1, 2)

        if interpretation.get("disease_profile"):
            names = list(interpretation["disease_profile"].keys())[:6]
            values = [
                interpretation["disease_profile"][n]["score"]
                for n in names
            ]

            plt.barh(names, values)
            plt.xlabel("Score", fontsize=9)
            plt.title("Disease Profile", fontsize=10)
        else:
            plt.text(
                0.5, 0.5,
                "No disease profile",
                ha="center", va="center",
                transform=plt.gca().transAxes
            )

        # ==================================
        # 3. Important Visits
        # ==================================
        plt.subplot(3, 1, 3)

        importance_map = {
            v["visit_index"] + 1: v["visit_importance"]
            for v in interpretation.get("important_visits", [])
        }
        y_importance = [importance_map.get(i, 0.0) for i in x_visits]

        plt.bar(x_visits, y_importance)
        plt.xticks(x_visits, fontsize=8)
        plt.ylabel("Alpha", fontsize=9)
        plt.xlabel("Visit", fontsize=9)
        plt.title("Important Visits", fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200)

        plt.show()
