"""
html_reporter.py
ایجاد گزارش HTML زیبا برای نتایج تفسیر
"""

import os
from datetime import datetime


class HTMLReporter:
    """Create HTML reports for patient interpretation results"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.html_dir = os.path.join(output_dir, 'html')
        os.makedirs(self.html_dir, exist_ok=True)

    def create_dashboard(self, interpretation, timestamp):
        """Create main clinical dashboard HTML"""
        patient_id = interpretation['patient_id']
        html_path = os.path.join(self.html_dir, 'clinical_dashboard.html')

        # Get top diseases
        top_diseases = list(interpretation['disease_profile'].items())[:5]
        clinical_insights = interpretation.get('clinical_insights', [])

        html_content = self._generate_html_content(
            patient_id, timestamp, interpretation,
            top_diseases, clinical_insights
        )

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html_path

    def _generate_html_content(self, patient_id, timestamp, interpretation,
                               top_diseases, clinical_insights):
        """Generate HTML content for the dashboard"""

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Clinical Dashboard - Patient {patient_id}</title>
    <style>
        {self._get_css_styles()}
    </style>
    <script>
        function toggleDetails(diseaseId) {{
            var element = document.getElementById(diseaseId);
            if (element.style.display === "none") {{
                element.style.display = "block";
            }} else {{
                element.style.display = "none";
            }}
        }}
    </script>
</head>
<body>
    <div class="container">
        {self._generate_header(patient_id, timestamp, interpretation)}

        <div class="grid">
            {self._generate_risk_assessment(interpretation)}
            {self._generate_disease_summary(interpretation)}
        </div>

        {self._generate_top_diseases(top_diseases)}
        {self._generate_clinical_insights(clinical_insights)}
        {self._generate_visualizations()}
        {self._generate_download_links()}
    </div>
</body>
</html>"""

    def _get_css_styles(self):
        """Return CSS styles for the dashboard"""
        return """
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header { 
            background: linear-gradient(135deg, #2c3e50, #4a6491); 
            color: white; 
            padding: 25px; 
            border-radius: 8px; 
            margin-bottom: 30px;
        }
        .risk-high { color: #e74c3c; font-weight: bold; }
        .risk-medium { color: #f39c12; font-weight: bold; }
        .risk-low { color: #27ae60; font-weight: bold; }
        .insight-critical { background: #ffebee; border-left: 4px solid #e74c3c; padding: 15px; margin: 10px 0; }
        .insight-high { background: #fff3e0; border-left: 4px solid #f39c12; padding: 15px; margin: 10px 0; }
        .insight-medium { background: #e8f4fd; border-left: 4px solid #3498db; padding: 15px; margin: 10px 0; }
        .disease-card { 
            background: #f8f9fa; 
            border: 1px solid #dee2e6; 
            border-radius: 8px; 
            padding: 15px; 
            margin: 10px 0;
            cursor: pointer;
            transition: all 0.3s;
        }
        .disease-card:hover {
            background: #e9ecef;
            transform: translateY(-2px);
        }
        .disease-details {
            display: none;
            background: white;
            padding: 15px;
            margin-top: 10px;
            border-left: 3px solid #6c757d;
        }
        .severity-severe { color: #dc3545; }
        .severity-moderate { color: #fd7e14; }
        .severity-mild { color: #28a745; }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
            margin: 20px 0;
        }
        .plot-container { 
            text-align: center; 
            margin: 20px 0; 
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        img { 
            max-width: 100%; 
            height: auto; 
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .download-links { 
            margin-top: 30px; 
            padding-top: 20px; 
            border-top: 1px solid #ddd; 
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 5px;
        }
        .badge-severe { background: #dc3545; color: white; }
        .badge-moderate { background: #fd7e14; color: white; }
        .badge-mild { background: #28a745; color: white; }
        .badge-chronic { background: #6f42c1; color: white; }
    """

    def _generate_header(self, patient_id, timestamp, interpretation):
        """Generate HTML header section"""
        return f"""
        <div class="header">
            <h1>📊 Clinical Dashboard</h1>
            <h2>Patient: {patient_id}</h2>
            <p>Analysis Date: {timestamp}</p>
            <p>Generated by Disease Interpreter v2.0</p>
        </div>"""

    def _generate_risk_assessment(self, interpretation):
        """Generate risk assessment section"""
        risk_class = f"risk-{interpretation['risk_category']}"
        return f"""
        <div style="background: #e8f4fd; padding: 20px; border-radius: 8px;">
            <h3>📈 Risk Assessment</h3>
            <p><strong>Mortality Risk:</strong> 
                <span class="{risk_class}">{interpretation['mortality_risk']:.2%}</span>
            </p>
            <p><strong>Category:</strong> 
                <span class="{risk_class}">{interpretation['risk_category'].upper()}</span>
            </p>
            <p><strong>Number of Visits:</strong> {interpretation['num_visits']}</p>
        </div>"""

    def _generate_disease_summary(self, interpretation):
        """Generate disease summary section"""
        chronic_count = sum(1 for d in interpretation['disease_profile'].values() if d['is_chronic'])
        severe_count = sum(1 for d in interpretation['disease_profile'].values() if d['severity'] == 'severe')

        return f"""
        <div style="background: #f0f8ff; padding: 20px; border-radius: 8px;">
            <h3>🏥 Disease Summary</h3>
            <p><strong>Total Diseases:</strong> {len(interpretation['disease_profile'])}</p>
            <p><strong>Severe Conditions:</strong> {severe_count}</p>
            <p><strong>Chronic Conditions:</strong> {chronic_count}</p>
            <p><strong>Clinical Insights:</strong> {len(interpretation['clinical_insights'])}</p>
        </div>"""

    def _generate_top_diseases(self, top_diseases):
        """Generate top diseases section"""
        diseases_html = ""
        for i, (disease, info) in enumerate(top_diseases, 1):
            disease_id = f"disease_{i}"
            badge_class = f"badge-{info['severity']}"

            diseases_html += f"""
        <div class="disease-card" onclick="toggleDetails('{disease_id}')">
            <h4>{i}. {disease}</h4>
            <p>
                <span class="badge {badge_class}">{info['severity'].upper()}</span>
                <strong>Score:</strong> {info['score']:.3f} | 
                <strong>Frequency:</strong> {info['frequency']} visits
                {'<span class="badge badge-chronic">CHRONIC</span>' if info['is_chronic'] else ''}
            </p>
            <div id="{disease_id}" class="disease-details">
                <p><strong>Raw Score:</strong> {info.get('raw_score', 0):.3f}</p>
                <p><strong>Severity Level:</strong> {info['severity']}</p>
                <p><strong>Visit Pattern:</strong> {'Chronic' if info['is_chronic'] else 'Acute'}</p>
            </div>
        </div>"""

        return f"""
        <h3>📋 Top Diseases</h3>
        {diseases_html}"""

    def _generate_clinical_insights(self, clinical_insights):
        """Generate clinical insights section"""
        if not clinical_insights:
            return "<h3>⚠️ Clinical Insights</h3><p>No clinical insights generated.</p>"

        insights_html = ""
        for insight in clinical_insights:
            priority_class = f"insight-{insight['priority']}"
            insights_html += f"""
        <div class="{priority_class}">
            <strong>[{insight['priority'].upper()}]</strong> {insight['message']}<br>
            <em>→ {insight['suggested_action']}</em>
        </div>"""

        return f"""
        <h3>⚠️ Clinical Insights</h3>
        {insights_html}"""

    def _generate_visualizations(self):
        """Generate visualizations section"""
        return """
        <h3>📊 Visualizations</h3>
        <div class="plot-container">
            <h4>Interpretation Summary</h4>
            <img src="../plots/interpretation_summary.png" alt="Interpretation Summary Plot">
        </div>

        <div class="plot-container">
            <h4>Disease Progression</h4>
            <img src="../plots/disease_progression.png" alt="Disease Progression Plot">
        </div>"""

    def _generate_download_links(self):
        """Generate download links section"""
        return """
        <div class="download-links">
            <h3>📥 Download Reports</h3>
            <p>
                <a href="../json/interpretation.json" download>📄 Complete JSON Data</a> | 
                <a href="../reports/clinical_report.txt" download>📝 Text Report</a> | 
                <a href="../json/clinical_insights.json" download>⚡ Clinical Insights (JSON)</a>
            </p>
            <p><small>All patient data is anonymized for privacy protection.</small></p>
        </div>"""