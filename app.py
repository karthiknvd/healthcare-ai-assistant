from flask import Flask, request, jsonify, render_template, session, url_for
import os
import requests
from datetime import datetime
import secrets
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# =============================================
# SET YOUR GEMINI API KEY HERE
# Get it FREE at: https://aistudio.google.com
# =============================================
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
# Or use environment variable (recommended):
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# =============================================

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

# Auto-create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Store conversation history per session
conversation_history = {}

SYSTEM_PROMPT = """You are a knowledgeable and empathetic Healthcare AI Assistant. You can answer ANY health-related question including:

- Symptom analysis and possible conditions
- Medication information (uses, side effects, interactions)
- Anatomy and physiology questions
- Disease explanations and mechanisms
- Mental health support and guidance
- Nutrition, diet, and wellness advice
- First aid and emergency guidance
- Medical terminology explanations
- Preventive care and healthy lifestyle tips
- Women's health, pediatric health, elderly care
- Lab results interpretation (general guidance)
- Medical procedures explained in simple terms

RESPONSE STYLE:
- Use emojis sparingly to make responses friendly (🏥 ⚕️ 💊 ⚠️ 💡 etc.)
- Use **bold** for important terms and key advice
- Be warm, empathetic, and non-judgmental
- Always include a short one-line disclaimer when giving personal medical advice

RESPONSE LENGTH — VERY IMPORTANT:
- Keep responses SHORT and CONCISE by default — 3 to 6 lines max for simple questions
- Only give detailed/long responses if the user explicitly asks: "explain more", "tell me more", "give details", "elaborate", "in detail" etc.
- Mention only the 2-3 most likely conditions, not every possibility
- Ask one follow-up question at most, not multiple
- Think like a knowledgeable friend giving a quick answer, not a medical textbook

SEVERITY AWARENESS:
- For HIGH severity symptoms (chest pain, difficulty breathing, stroke signs, severe bleeding): immediately emphasize emergency care
- For MODERATE symptoms: recommend seeing a doctor within 24-48 hours
- For MILD symptoms: suggest home care with monitoring

IMPORTANT RULES:
- Never refuse health questions — always provide helpful, accurate information
- Only add a disclaimer when: user asks about specific medications/dosages, symptoms suggest a serious/emergency condition, or the topic involves surgery/procedures. For general health questions, do NOT add any disclaimer.
- If someone mentions a life-threatening emergency, prioritize emergency services info first
- For mental health crises, provide crisis resources
- You have memory of the conversation, so reference previous symptoms or context when relevant"""


def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(16)
    return session['session_id']


def call_gemini_api(session_id, user_message):
    """Call the Gemini API with full conversation history"""

    if session_id not in conversation_history:
        conversation_history[session_id] = []

    # Add user message to history
    conversation_history[session_id].append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    # Keep last 20 messages to avoid token limit issues
    messages = conversation_history[session_id][-20:]

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": messages,
        "generationConfig": {
            "maxOutputTokens": 2048,
            "temperature": 0.7
        }
    }

    try:
        response = requests.post(GEMINI_URL, json=payload, timeout=30)
        data = response.json()

        if response.status_code == 200:
            candidate = data["candidates"][0]
            finish_reason = candidate.get("finishReason", "STOP")
            reply = candidate["content"]["parts"][0]["text"]

            if finish_reason == "MAX_TOKENS":
                reply += "\n\n⚠️ *(Response was long — ask me to continue if needed.)*"

            conversation_history[session_id].append({
                "role": "model",
                "parts": [{"text": reply}]
            })

            return markdown_to_html(reply)

        elif response.status_code == 400:
            error_msg = data.get("error", {}).get("message", "Bad request")
            return f"❌ <b>API Error:</b> {error_msg}"

        elif response.status_code == 403:
            return "❌ <b>Invalid API Key.</b> Please check your Gemini API key in app.py.<br>Get a free key at <b>aistudio.google.com</b>"

        elif response.status_code == 429:
            return "⏳ <b>Rate limit reached.</b> You've hit the free tier limit. Please wait a minute and try again."

        else:
            error_msg = data.get("error", {}).get("message", "Unknown error")
            return f"❌ <b>API Error {response.status_code}:</b> {error_msg}"

    except requests.exceptions.Timeout:
        return "⏱️ <b>Request timed out.</b> Please try again."
    except requests.exceptions.ConnectionError:
        return "🔌 <b>Connection error.</b> Please check your internet connection."
    except KeyError:
        return "❌ <b>Unexpected response format from Gemini.</b> Please try again."
    except Exception as e:
        return f"❌ <b>Unexpected error:</b> {str(e)}"


def markdown_to_html(text):
    """Convert markdown formatting to HTML for chat display"""
    import re

    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*([^*\n]+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'^### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$',  r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$',   r'<h2>\1</h2>', text, flags=re.MULTILINE)

    lines = text.split('\n')
    result = []

    i = 0
    while i < len(lines):
        line = lines[i]

        ul_match = re.match(r'^(\s*)[-*•] (.+)$', line)
        ol_match = re.match(r'^(\s*)\d+\. (.+)$', line)

        if ul_match or ol_match:
            block_lines = []
            j = i
            while j < len(lines):
                l = lines[j]
                is_ul = re.match(r'^(\s*)[-*•] (.+)$', l)
                is_ol = re.match(r'^(\s*)\d+\. (.+)$', l)
                is_blank = l.strip() == ''

                if is_ul or is_ol:
                    block_lines.append(l)
                    j += 1
                elif is_blank:
                    k = j + 1
                    while k < len(lines) and lines[k].strip() == '':
                        k += 1
                    if k < len(lines) and (
                        re.match(r'^(\s*)[-*•] (.+)$', lines[k]) or
                        re.match(r'^(\s*)\d+\. (.+)$', lines[k])
                    ):
                        j = k
                    else:
                        break
                else:
                    break

            list_stack = []

            def close_to(target_indent):
                while list_stack and list_stack[-1][1] > target_indent:
                    tag, _ = list_stack.pop()
                    result.append(f'</{tag}>')

            def close_all():
                while list_stack:
                    tag, _ = list_stack.pop()
                    result.append(f'</{tag}>')

            ol_counters = {}

            for bl in block_lines:
                bul = re.match(r'^(\s*)[-*•] (.+)$', bl)
                bol = re.match(r'^(\s*)\d+\. (.+)$', bl)
                if bul:
                    indent = len(bul.group(1))
                    content = bul.group(2)
                    close_to(indent)
                    if not list_stack or list_stack[-1][1] < indent:
                        list_stack.append(('ul', indent))
                        result.append('<ul>')
                    elif list_stack[-1][0] != 'ul':
                        tag, _ = list_stack.pop()
                        result.append(f'</{tag}>')
                        list_stack.append(('ul', indent))
                        result.append('<ul>')
                    result.append(f'<li>{content}</li>')
                elif bol:
                    indent = len(bol.group(1))
                    content = bol.group(2)
                    close_to(indent)
                    if not list_stack or list_stack[-1][1] < indent:
                        list_stack.append(('ol', indent))
                        result.append('<ol>')
                    elif list_stack[-1][0] != 'ol':
                        tag, _ = list_stack.pop()
                        result.append(f'</{tag}>')
                        list_stack.append(('ol', indent))
                        result.append('<ol>')
                    ol_counters[indent] = ol_counters.get(indent, 0) + 1
                    n = ol_counters[indent]
                    result.append(f'<li data-n="{n}">{content}</li>')

            close_all()
            i = j

        else:
            if line.strip() == '':
                result.append('<br>')
            else:
                result.append(line)
            i += 1

    text = '\n'.join(result)
    text = re.sub(r'(<br>\s*){2,}', '<br>', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(
        r'<br>---<br>',
        '<hr style="border:none;border-top:1px solid rgba(255,255,255,0.12);margin:10px 0">',
        text
    )
    text = re.sub(
        r'`(.+?)`',
        r'<code style="background:rgba(255,255,255,0.1);padding:2px 6px;border-radius:4px;font-family:monospace;font-size:0.9em">\1</code>',
        text
    )

    return text


# ----------------------------
# IMAGE PROCESSING
# ----------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def generate_image_descriptions(img, gray, edges, enhanced, valid_contours, h_img, w_img):
    """
    Generate pure Python/OpenCV-based descriptions for each processed image.
    No AI API involved — all stats computed from pixel data.
    """
    image_area = h_img * w_img

    # ── 1. Edge Map ──────────────────────────────────────────────────────────
    edge_pixel_count = int(np.count_nonzero(edges))
    edge_density = round(edge_pixel_count / image_area * 100, 1)

    if edge_density < 3:
        complexity = "very smooth with minimal detectable boundaries"
        complexity_note = "Likely a low-contrast or uniform image — edge-based analysis has limited utility here."
    elif edge_density < 8:
        complexity = "moderately structured with clear boundaries"
        complexity_note = "Good edge definition. Key anatomical borders are identifiable."
    elif edge_density < 18:
        complexity = "highly detailed with dense structural features"
        complexity_note = "Rich in edges — may indicate textured tissue or multiple overlapping structures."
    else:
        complexity = "extremely complex with very high edge density"
        complexity_note = "Very high edge count — image may be noisy or contain fine-grained textures."

    edge_desc = (
        f"<b>Edge pixels detected:</b> {edge_pixel_count:,} ({edge_density}% of image area)<br>"
        f"The image is <b>{complexity}</b>.<br>"
        f"{complexity_note}"
    )

    # ── 2. Contrast Enhanced ────────────────────────────────────────────────
    orig_mean = round(float(np.mean(gray)), 1)
    orig_std  = round(float(np.std(gray)), 1)
    enh_mean  = round(float(np.mean(enhanced)), 1)
    enh_std   = round(float(np.std(enhanced)), 1)
    improvement = round((enh_std - orig_std) / max(orig_std, 1) * 100, 1)

    orig_min, orig_max = int(gray.min()), int(gray.max())
    enh_min,  enh_max  = int(enhanced.min()), int(enhanced.max())

    brightness_shift = "increased" if enh_mean > orig_mean else "slightly reduced"

    contrast_desc = (
        f"<b>Contrast improvement:</b> +{improvement}% (std dev {orig_std} → {enh_std})<br>"
        f"<b>Pixel range:</b> [{orig_min}–{orig_max}] → [{enh_min}–{enh_max}] &nbsp;|&nbsp; "
        f"<b>Brightness:</b> {brightness_shift} ({orig_mean} → {enh_mean})<br>"
        f"CLAHE (Clip=3.0, 8×8 grid) redistributed local histograms, making subtle tissue "
        f"variations more distinguishable without over-brightening bright regions."
    )

    # ── 3. Attention Heatmap ─────────────────────────────────────────────────
    flat = gray.flatten().astype(np.float32)
    high_pct = round(float(np.sum(flat > 200) / image_area * 100), 1)
    mid_pct  = round(float(np.sum((flat >= 80) & (flat <= 200)) / image_area * 100), 1)
    low_pct  = round(max(0.0, 100 - high_pct - mid_pct), 1)

    # Find centroid of brightest region (potential area of interest)
    bright_mask = (gray > 180).astype(np.uint8)
    moments = cv2.moments(bright_mask)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        centroid_note = f"Brightest region centred around pixel ({cx}, {cy})."
    else:
        centroid_note = "No dominant bright region detected."

    heatmap_desc = (
        f"<b>Intensity zones:</b> "
        f"🔴 High (>200): {high_pct}% &nbsp;|&nbsp; "
        f"🟡 Mid (80–200): {mid_pct}% &nbsp;|&nbsp; "
        f"🔵 Low (<80): {low_pct}%<br>"
        f"{centroid_note}<br>"
        f"Warm colours (red/yellow) = denser/brighter tissue. "
        f"Cool colours (blue) = less dense or air-filled regions."
    )

    # ── 4. Anomaly Detection ─────────────────────────────────────────────────
    n_rois = len(valid_contours)
    if n_rois == 0:
        anomaly_desc = (
            "<b>No regions of interest flagged.</b><br>"
            "The image appears structurally uniform — no areas exceeded the anomaly "
            "threshold (min 500px², max 40% of image). This may indicate a clean scan "
            "or an image that requires different preprocessing parameters."
        )
    else:
        roi_lines = []
        for idx, c in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(c)
            area = int(cv2.contourArea(c))
            aspect = round(w / max(h, 1), 2)
            shape_hint = "roughly square" if 0.7 < aspect < 1.4 else ("wider than tall" if aspect >= 1.4 else "taller than wide")
            roi_lines.append(
                f"<b>ROI {idx+1}:</b> {w}×{h}px at position ({x}, {y}) — "
                f"area {area:,}px², {shape_hint}"
            )
        total_roi_area = sum(int(cv2.contourArea(c)) for c in valid_contours)
        coverage = round(total_roi_area / image_area * 100, 1)
        anomaly_desc = (
            f"<b>{n_rois} region(s) of interest detected</b> (covering {coverage}% of image):<br>"
            "<br>".join(roi_lines) + "<br>"
            "Detected via adaptive Gaussian thresholding + morphological opening. "
            "Red bounding boxes mark candidate regions for closer inspection."
        )

    return {
        "edge_map":          edge_desc,
        "contrast_enhanced": contrast_desc,
        "attention_heatmap": heatmap_desc,
        "anomaly_detection": anomaly_desc,
    }


def process_medical_image(image_path):
    """Process medical image and generate 4 visualizations using OpenCV"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to read image")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"processed_{timestamp}"
        processed_images = {}

        # 1. Structural Edge Map (Canny Edge Detection)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edge_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{base_name}_edges.png")
        cv2.imwrite(edge_path, edges_colored)
        processed_images['edge_map'] = url_for('static', filename=f'processed/{base_name}_edges.png')

        # 2. Contrast Enhanced Image (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_colored = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        enhanced_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{base_name}_enhanced.png")
        cv2.imwrite(enhanced_path, enhanced_colored)
        processed_images['contrast_enhanced'] = url_for('static', filename=f'processed/{base_name}_enhanced.png')

        # 3. AI Attention Heatmap (JET colormap overlay)
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        heatmap_blended = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
        heatmap_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{base_name}_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap_blended)
        processed_images['attention_heatmap'] = url_for('static', filename=f'processed/{base_name}_heatmap.png')

        # 4. Anomaly Detection (Adaptive threshold + contour detection)
        anomaly_img = img.copy()
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 5
        )
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_area = h_img * w_img
        valid_contours = [c for c in contours if 500 < cv2.contourArea(c) < 0.4 * image_area]
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:3]

        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 20 or h < 20:
                continue
            cv2.rectangle(anomaly_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(anomaly_img, 'AI ROI', (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        anomaly_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{base_name}_anomaly.png")
        cv2.imwrite(anomaly_path, anomaly_img)
        processed_images['anomaly_detection'] = url_for('static', filename=f'processed/{base_name}_anomaly.png')

        # Generate Python-computed descriptions (no API call)
        descriptions = generate_image_descriptions(
            img, gray, edges, enhanced, valid_contours, h_img, w_img
        )

        return processed_images, descriptions

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, None


# ----------------------------
# ROUTES
# ----------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'reply': 'Please enter a message.'})

    session_id = get_session_id()
    message_lower = message.lower()

    reset_commands = ['reset', 'clear', 'start over', 'new symptoms', 'forget']
    if any(cmd in message_lower for cmd in reset_commands) and len(message.split()) <= 4:
        if session_id in conversation_history:
            del conversation_history[session_id]
        return jsonify({'reply': '🔄 <b>Conversation reset.</b> How can I help you today?'})

    reply = call_gemini_api(session_id, message)
    return jsonify({'reply': reply})


@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """Handle medical image upload and OpenCV processing"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, BMP, or TIFF.'}), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        processed_images, descriptions = process_medical_image(filepath)

        if processed_images is None:
            return jsonify({'error': 'Failed to process image'}), 500

        return jsonify({
            'success': True,
            'original_image': url_for('static', filename=f'uploads/{unique_filename}'),
            'processed_images': processed_images,
            'descriptions': descriptions          # ← NEW: per-image descriptions
        })

    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


# ----------------------------
# START SERVER
# ----------------------------
if __name__ == "__main__":
    if not GEMINI_API_KEY or GEMINI_API_KEY == "AIza-YOUR_KEY_HERE":
        print("\n" + "=" * 60)
        print("⚠️  WARNING: No Gemini API key set!")
        print("   Edit app.py and set your GEMINI_API_KEY")
        print("   Get a FREE key at: https://aistudio.google.com")
        print("=" * 60 + "\n")
    else:
        print("✅ Gemini API key found. Starting server...")

    app.run(debug=True, host="0.0.0.0", port=5000)