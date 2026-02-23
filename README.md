# 🏥 **Healthcare AI Assistant**

*A modern healthcare information chatbot with medical image analysis and a fully custom UI.*

This project delivers:

* Smart health-related responses
* Symptom guidance and explanations
* Medication, wellness, and general health information
* Local-only medical image processing (OpenCV)
* A clean, animated chat UI
* Friendly, short, and safety-aware answers

---

## ⭐ **Features**

### 🗣️ **1. Health Chat Assistant (uses the Gemini 2.5 Flash model)**

* Short and clear answers
* Symptom understanding
* Medication info & safety notes
* Disease explanations
* Nutrition & mental health guidance
* Severity-aware responses
* Remembers last 20 messages
* Disclaimers when appropriate

### 🖼️ **2. Local Medical Image Processing (Offline)**

Using **OpenCV + NumPy**:

* Structural edge detection
* Contrast enhancement (CLAHE)
* Heatmap visualization
* Basic anomaly ROI detection
* Pure Python descriptions based on pixel analysis

### 🧠 **3. Modern Chat UI**

* Typing animation
* Smooth transitions
* Quick chips and example prompts
* Auto scroll system
* Mode toggle (Chat ↔ Image Analysis)
* Clean dark theme

### 🔒 **4. Privacy-Friendly**

* Images never leave your device
* Only text is sent to the AI model
* No cloud-based image processing
* No logs stored externally

---

## 🏗 **Project Structure**

```text
healthcare_ai/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── static/
│   ├── style.css           # Main stylesheet
│   ├── script.js           # Frontend JavaScript
│   ├── uploads/            # All uploaded medical images
│   └── processed/          # All processed output images
│
└── templates/
    └── index.html          # Full frontend UI
```

---

## ⚙️ **Installation & Setup**

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/karthiknvd/healthcare-ai-assistant.git
cd healthcare-ai-assistant
```

### **2️⃣ Create a Virtual Environment**

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### **3️⃣ Install Requirements**

```bash
pip install -r requirements.txt
```

### **4️⃣ Add Your API Key**

Open `app.py` and replace:

```python
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
```

### **5️⃣ Start the App**

```bash
python app.py
```

Visit:
👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 🛠️ **Tech Stack**

* Python (Flask)
* OpenCV
* NumPy
* HTML / CSS / JavaScript
* **Gemini 2.5 Flash model**

---

## 📡 **API Endpoints**

### **POST `/ask`**

Handles user chat messages and returns a structured health response.

### **POST `/analyze-image`**

Processes medical images locally and returns:

* Edge map
* Enhanced image
* Heatmap
* Anomaly map
* Descriptions

---

## 🔍 **Image Analysis Logic**

All processed **locally** using OpenCV:

* **Edge Map:** Canny
* **Enhanced Image:** CLAHE
* **Heatmap:** JET colormap
* **Anomalies:** Adaptive threshold + contours
* **Descriptions:** Based on intensity, contrast, edge density, ROI size

---

## 💬 **Chat Behavior**

The assistant is designed to:

* Give short, direct answers
* Expand only when asked
* Detect severity
* Speak in a friendly tone
* Add disclaimers for risky topics
* Maintain session-based memory

---

## 🧹 **.gitignore**

```gitignore
.venv/
app.log
*.log

static/uploads/*
!static/uploads/.gitkeep

static/processed/*
!static/processed/.gitkeep
```

---

## ⚠️ **Medical Disclaimer**

This project is for **informational purposes only** and does **not** provide medical diagnosis.
Always consult a healthcare professional for medical decisions.