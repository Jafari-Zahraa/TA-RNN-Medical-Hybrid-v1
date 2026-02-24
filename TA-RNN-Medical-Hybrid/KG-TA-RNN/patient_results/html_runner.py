import os
import glob
import webbrowser
import subprocess

# -----------------------------------
# Loop until valid patient number is entered
# -----------------------------------
while True:
    patient_num = input("Enter patient number: ").strip()

    if not patient_num.isdigit():
        print("❌ Invalid input. Please enter a numeric patient number.")
        continue

    # Build dynamic HTML path
    html_path = f"PATIENT_{patient_num}/html/clinical_dashboard.html"

    # Check if file exists
    html_files = glob.glob(html_path, recursive=True)
    if not html_files:
        print(f"❌ No data found for patient {patient_num}. Please try again.")
        continue

    # Valid file found, break the loop
    html_file = max(html_files, key=os.path.getmtime)
    print(f"✅ HTML file found: {html_file}")
    break

# -----------------------------------
# Basic file validation
# -----------------------------------
file_size = os.path.getsize(html_file)
print(f"📏 File size: {file_size} bytes")

if file_size < 100:
    print("⚠️ Warning: HTML file is very small and may be empty.")
    with open(html_file, "r", encoding="utf-8") as f:
        print("\n📄 File preview (first 500 characters):")
        print(f.read(500))

# -----------------------------------
# Open HTML file
# -----------------------------------
abs_path = os.path.abspath(html_file)
file_url = f"file:///{abs_path}"

print("\n🚀 Opening HTML file...")

# Method 1: Default web browser
try:
    webbrowser.open(file_url)
    print("✅ Opened with webbrowser")
except Exception as e:
    print(f"❌ webbrowser failed: {e}")

# Method 2: OS-specific fallback (Windows)
try:
    os.startfile(html_file)
    print("✅ Opened with os.startfile")
except Exception:
    pass

# Method 3: Subprocess fallback
try:
    subprocess.Popen(["start", html_file], shell=True)
    print("✅ Opened with subprocess")
except Exception:
    pass
