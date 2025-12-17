import os
import json
import time
import math
import re
import sys
import pandas as pd
import numpy as np
import recurring_ical_events
from datetime import datetime, timedelta, time as dt_time
from typing import List, Optional
from zoneinfo import ZoneInfo
from collections import defaultdict

# Web Framework & Utils
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from whitenoise import WhiteNoise

# Data & AI Models
from sklearn.linear_model import ElasticNet
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Calendar Processing
from icalendar import Calendar as ICalLoader, Event as IcsEvent
from icalendar import Calendar 

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
CSV_PATH = os.path.join(BASE_DIR, 'survey.csv')

# Initialize App
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)

# WhiteNoise (Static file serving)
app.wsgi_app = WhiteNoise(app.wsgi_app, root=STATIC_DIR, prefix='static/')

# File System Setup
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB limit

# Constants
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LOCAL_TZ = ZoneInfo("America/New_York")
CHUNK_SIZE = 30 # Matching Notebook Block 5

# ==========================================
# 1. CORE MATH HELPERS (PORTED FROM NOTEBOOK BLOCK 3)
# ==========================================

def _to_local_dt(value, local_tz: ZoneInfo):
    if value is None: return None
    if isinstance(value, datetime):
        if value.tzinfo is None: return value.replace(tzinfo=local_tz)
        return value.astimezone(local_tz)
    # Handle dates (all-day events)
    return datetime.combine(value, dt_time.min, tzinfo=local_tz)

def clip_blocks_to_horizon(blocks, horizon_start: datetime, horizon_end: datetime):
    clipped = []
    for start, end in blocks:
        if end <= horizon_start or start >= horizon_end: continue
        s = max(start, horizon_start)
        e = min(end, horizon_end)
        if s < e: clipped.append((s, e))
    return clipped

def merge_busy_blocks(blocks, join_touching: bool = True):
    if not blocks: return []
    blocks = sorted(blocks, key=lambda x: x[0])
    merged = []
    cur_start, cur_end = blocks[0]
    for s, e in blocks[1:]:
        if (join_touching and s <= cur_end) or (not join_touching and s < cur_end):
            if e > cur_end: cur_end = e
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))
    return merged

def subtract_busy_from_window(window_start, window_end, busy_blocks):
    relevant = []
    for s, e in busy_blocks:
        if e <= window_start or s >= window_end: continue
        s_clipped = max(s, window_start)
        e_clipped = min(e, window_end)
        if s_clipped < e_clipped: relevant.append((s_clipped, e_clipped))
    relevant.sort(key=lambda x: x[0])
    free = []
    cur = window_start
    for s, e in relevant:
        if s > cur: free.append((cur, s))
        cur = max(cur, e)
    if cur < window_end: free.append((cur, window_end))
    return free

def add_buffer_to_busy_timeline(busy_timeline, buffer_minutes=15):
    delta = timedelta(minutes=buffer_minutes)
    buffered = [(s - delta, e + delta) for (s, e) in busy_timeline]
    return merge_busy_blocks(buffered, join_touching=True)

def build_free_blocks(WORK_WINDOWS, BUSY_TIMELINE, horizon_start, horizon_end):
    FREE_BLOCKS = {}
    current_date = horizon_start.date()
    now = datetime.now(LOCAL_TZ)

    while current_date <= horizon_end.date():
        day_start_cal = datetime.combine(current_date, dt_time.min).replace(tzinfo=LOCAL_TZ)
        day_end_cal = day_start_cal + timedelta(days=1)

        # Clip start to NOW so we don't book in the past
        effective_start = max(day_start_cal, horizon_start)
        if current_date == now.date():
            effective_start = max(effective_start, now)

        current_day_start = effective_start
        current_day_end = min(day_end_cal, horizon_end)

        if current_day_start >= current_day_end:
            current_date += timedelta(days=1); continue

        weekday = current_date.weekday()
        day_work_windows = WORK_WINDOWS.get(weekday, [])

        day_busy = []
        for s, e in BUSY_TIMELINE:
            if e <= current_day_start or s >= current_day_end: continue
            s_clipped = max(s, current_day_start)
            e_clipped = min(e, current_day_end)
            if s_clipped < e_clipped: day_busy.append((s_clipped, e_clipped))
        day_busy.sort(key=lambda x: x[0])

        day_free = []
        for start_hour, end_hour in day_work_windows:
            w_start_cal = day_start_cal + timedelta(hours=float(start_hour))
            w_end_cal = day_start_cal + timedelta(hours=float(end_hour))

            w_actual_start = max(w_start_cal, current_day_start)
            w_actual_end = min(w_end_cal, current_day_end)

            if w_actual_start < w_actual_end:
                day_free.extend(subtract_busy_from_window(w_actual_start, w_actual_end, day_busy))

        FREE_BLOCKS[current_date] = day_free
        current_date += timedelta(days=1)
    return FREE_BLOCKS

# ==========================================
# 2. SCHEDULING LOGIC (PORTED FROM NOTEBOOK BLOCK 4.5)
# ==========================================

def generate_sessions_from_courses(courses, default_session_minutes=30):
    sessions = []
    for i, c in enumerate(courses):
        # MAPPING: Frontend sends 'predicted_hours', Notebook Logic used 'time_spent_hours'
        total_time_mins = c.get("predicted_hours", 2.0) * 60
        
        num_sessions = math.ceil(total_time_mins / default_session_minutes)
        if num_sessions == 0: continue

        base_duration = int(total_time_mins // num_sessions)
        remainder = int(total_time_mins % num_sessions)

        # Parse due date
        try:
            due_str = c.get("date")
            # Assume end of day for the due date
            due_dt = datetime.strptime(due_str, '%Y-%m-%d').replace(hour=23, minute=59).replace(tzinfo=LOCAL_TZ)
        except:
            continue

        for j in range(num_sessions):
            dur = base_duration + (1 if j < remainder else 0)
            sessions.append({
                "assignment_id": f"{i}_{j}",
                "assignment_name": c.get("name"),
                "class_name": c.get("type"), # Using 'type' as class name proxy for now
                "duration_minutes": dur,
                "due_date": due_dt
            })
            
    # Sort by due date (Earliest Deadline First)
    return sorted(sessions, key=lambda x: x["due_date"])

def schedule_sessions_load_balanced(free_blocks_map, sessions, max_hours_per_day=24):
    scheduled = []
    unscheduled = []
    daily_usage_minutes = defaultdict(float)
    max_minutes = max_hours_per_day * 60

    # Deep copy map to avoid modifying original during iteration
    free_map = {k: v[:] for k, v in free_blocks_map.items()}
    sorted_dates = sorted(free_map.keys())

    for session in sessions:
        placed = False
        duration_mins = session["duration_minutes"]
        duration = timedelta(minutes=duration_mins)

        for d in sorted_dates:
            # Don't schedule after due date
            if d > session["due_date"].date(): break
            
            # Load Balancing Check
            if daily_usage_minutes[d] + duration_mins > max_minutes: continue

            day_blocks = free_map[d]
            for i, (start, end) in enumerate(day_blocks):
                block_duration = end - start
                if block_duration >= duration:
                    session_start = start
                    session_end = start + duration

                    rec = session.copy()
                    rec["start"] = session_start
                    rec["end"] = session_end
                    scheduled.append(rec)

                    daily_usage_minutes[d] += duration_mins

                    # Update free block in map
                    new_start = session_end
                    if new_start < end:
                        free_map[d][i] = (new_start, end)
                    else:
                        free_map[d].pop(i)

                    placed = True
                    break
            if placed: break

        if not placed:
            unscheduled.append(session)

    return scheduled, unscheduled

def merge_contiguous_sessions(scheduled_sessions):
    if not scheduled_sessions: return []
    # Sort by start time
    sorted_sessions = sorted(scheduled_sessions, key=lambda x: x["start"])
    merged = []
    current_block = sorted_sessions[0]

    for next_block in sorted_sessions[1:]:
        same_assignment = (current_block["assignment_name"] == next_block["assignment_name"])
        touching_time = (current_block["end"] == next_block["start"])

        if same_assignment and touching_time:
            # Extend current block
            current_block["end"] = next_block["end"]
            current_block["duration_minutes"] += next_block["duration_minutes"]
        else:
            merged.append(current_block)
            current_block = next_block
    merged.append(current_block)
    return merged

# ==========================================
# 3. MAIN LOGIC CONTROLLER
# ==========================================

def parse_ics_to_list(ics_path, start, end):
    """Parses uploaded ICS into simple list of (start, end) tuples."""
    if not ics_path or not os.path.exists(ics_path):
        return []
    try:
        with open(ics_path, 'rb') as f:
            cal = ICalLoader.from_ical(f.read())
        occurrences = recurring_ical_events.of(cal).between(start, end)
        blocks = []
        for comp in occurrences:
            dtstart = _to_local_dt(comp.get("DTSTART").dt, LOCAL_TZ)
            dtend = comp.get("DTEND")
            if dtend:
                dtend = _to_local_dt(dtend.dt, LOCAL_TZ)
            else:
                dtend = dtstart + timedelta(hours=1)
            blocks.append((dtstart, dtend))
        return blocks
    except Exception as e:
        print(f"ICS Error: {e}")
        return []

def run_scheduler_logic(courses, preferences, user_ics_path, output_filename):
    now = datetime.now(LOCAL_TZ)
    # 1. Determine Horizon (e.g., 60 days out)
    horizon_start = now
    horizon_end = now + timedelta(days=60)

    # 2. Parse Work Windows (Frontend Strings -> Float Tuples)
    # Format: "09:00" -> 9.0
    def time_str_to_float(t_str):
        h, m = map(int, t_str.split(':'))
        return h + (m/60)

    wd_start = time_str_to_float(preferences.get('weekdayStart', '09:00'))
    wd_end = time_str_to_float(preferences.get('weekdayEnd', '21:00'))
    we_start = time_str_to_float(preferences.get('weekendStart', '10:00'))
    we_end = time_str_to_float(preferences.get('weekendEnd', '20:00'))

    # Dictionary: 0-4 (Mon-Fri), 5-6 (Sat-Sun)
    WORK_WINDOWS = {}
    for i in range(7):
        WORK_WINDOWS[i] = [(wd_start, wd_end)] if i < 5 else [(we_start, we_end)]

    # 3. Parse Busy Timeline
    raw_busy = parse_ics_to_list(user_ics_path, horizon_start, horizon_end)
    clipped_busy = clip_blocks_to_horizon(raw_busy, horizon_start, horizon_end)
    BUSY_TIMELINE = merge_busy_blocks(clipped_busy, join_touching=True)
    
    # Optional: Add buffer (15 min)
    BUSY_TIMELINE = add_buffer_to_busy_timeline(BUSY_TIMELINE, buffer_minutes=15)

    # 4. Build Free Blocks
    FREE_BLOCKS = build_free_blocks(WORK_WINDOWS, BUSY_TIMELINE, horizon_start, horizon_end)

    # 5. Generate Sessions
    all_sessions = generate_sessions_from_courses(courses, default_session_minutes=CHUNK_SIZE)

    # 6. Run Schedule
    scheduled, unscheduled = schedule_sessions_load_balanced(FREE_BLOCKS, all_sessions)

    # 7. Merge and Export
    final_blocks = merge_contiguous_sessions(scheduled)

    cal = Calendar()
    cal.add('prodid', '-//StudentOS//mxm.dk//')
    cal.add('version', '2.0')
    cal.add('x-wr-calname', "StudentOS Schedule")

    for row in final_blocks:
        event = IcsEvent()
        event.add('summary', f"{row['class_name']}: {row['assignment_name']}")
        event.add('dtstart', row['start'])
        event.add('dtend', row['end'])
        event.add('description', f"Work on {row['assignment_name']}. Duration: {row['duration_minutes']} min.")
        event.add('uid', f"{row['assignment_id']}_{row['start'].strftime('%Y%m%dT%H%M%S')}@studentos")
        cal.add_component(event)
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    with open(output_path, 'wb') as f:
        f.write(cal.to_ical())
    
    return output_filename

# ==========================================
# 4. PDF & ML UTILS (UNCHANGED)
# ==========================================

def parse_syllabus(file_path):
    if not genai or not GEMINI_API_KEY: return None
    try:
        class AssignmentItem(BaseModel):
            date: str = Field(description="YYYY-MM-DD")
            time: Optional[str] = Field(description="Deadline time or null")
            assignment_name: str = Field(description="Name")
            category: str = Field(description="Category")
            description: str = Field(description="Details")

        class SyllabusResponse(BaseModel):
            metadata: dict = Field(description="Metadata")
            assignments: List[AssignmentItem]

        client = genai.Client(api_key=GEMINI_API_KEY)
        file_upload = client.files.upload(file=file_path)
        
        while file_upload.state.name == "PROCESSING":
            time.sleep(1)
            file_upload = client.files.get(name=file_upload.name)
        
        if file_upload.state.name != "ACTIVE": return None

        prompt = "Extract assignments (Dates YYYY-MM-DD), readings, and deliverables."
        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=[file_upload, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=SyllabusResponse)
        )
        data = response.parsed
        rows = []
        c_name = getattr(data.metadata, 'course_name', 'Parsed Course')
        for item in data.assignments:
            rows.append({"Course": c_name, "Date": item.date, "Time": item.time, "Category": item.category, "Assignment": item.assignment_name, "Description": item.description})
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Parsing error: {e}")
        return None

def map_pdf_category(cat):
    c = str(cat).lower()
    if 'reading' in c: return 'readings'
    if 'writing' in c: return 'essay'
    if 'exam' in c: return 'p_set'
    if 'project' in c: return 'research_paper'
    return 'p_set'

model = None
model_columns = []

def initialize_model():
    global model, model_columns
    if not ElasticNet or not os.path.exists(CSV_PATH): 
        print("Model initialization skipped.")
        return

    try:
        df = pd.read_csv(CSV_PATH)
        df = df.rename(columns={
            'What year are you? ': 'year', 
            'What is your major/concentration?': 'major', 
            'What type of assignment was it?': 'assignment_type', 
            'Approximately how long did it take (in hours)': 'time_spent_hours'
        })
        categorical_cols = ['year', 'assignment_type', 'external_resources', 'work_location', 'worked_in_group', 'submitted_in_person']
        for col in categorical_cols:
            if col in df.columns: 
                df = pd.get_dummies(df, columns=[col], prefix=col, dtype=int, drop_first=True)
        df = df.select_dtypes(include=[np.number])
        if 'time_spent_hours' in df.columns:
            X = df.drop('time_spent_hours', axis=1)
            y = df['time_spent_hours']
            clf = ElasticNet(alpha=0.078, l1_ratio=0.95, max_iter=5000)
            clf.fit(X, y)
            model = clf
            model_columns = list(X.columns)
            print("✅ Model Trained successfully.")
    except Exception as e:
        print(f"Training Failed: {e}")

initialize_model()

# ==========================================
# 5. ROUTES
# ==========================================

@app.route('/', methods=['GET'])
def home():
    return render_template('mains.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

@app.route('/api/generate-schedule', methods=['POST'])
def generate_schedule():
    try:
        data_str = request.form.get('data')
        if not data_str: return jsonify({'error': 'No data provided'}), 400
        
        req_data = json.loads(data_str)
        survey = req_data.get('survey', {})
        courses = req_data.get('courses', [])
        preferences = req_data.get('preferences', {})

        # PDF Uploads
        uploaded_pdfs = request.files.getlist('pdfs')
        if uploaded_pdfs:
            for pdf in uploaded_pdfs:
                if pdf.filename == '': continue
                filename = secure_filename(pdf.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                pdf.save(path)
                df_parsed = parse_syllabus(path)
                if df_parsed is not None and not df_parsed.empty:
                    for _, row in df_parsed.iterrows():
                        courses.append({
                            'name': f"{row['Course']}: {row['Assignment']}",
                            'type': map_pdf_category(row['Category']),
                            'date': row['Date']
                        })

        # ICS Upload
        user_ics_path = None
        ics_file = request.files.get('ics')
        if ics_file and ics_file.filename != '':
            filename = secure_filename(ics_file.filename)
            user_ics_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            ics_file.save(user_ics_path)

        # ML Prediction
        for c in courses:
            if model and model_columns:
                input_data = {col: 0 for col in model_columns}
                y_col = f"year_{survey.get('year')}"
                if y_col in input_data: input_data[y_col] = 1
                t_col = f"assignment_type_{c.get('type')}"
                if t_col in input_data: input_data[t_col] = 1
                pred = model.predict(pd.DataFrame([input_data]))[0]
                c['predicted_hours'] = round(max(0.5, pred), 1)
            else:
                c['predicted_hours'] = 2.0

        # Run The REAL Scheduler
        output_filename = f"schedule_{int(time.time())}.ics"
        result_file = run_scheduler_logic(courses, preferences, user_ics_path, output_filename)

        return jsonify({
            'message': 'Success',
            'courses': courses,
            'ics_url': f"/download/{result_file}"
        })

    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
