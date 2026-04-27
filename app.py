import cv2
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from datetime import datetime

app = Flask(__name__)

DATASET_PATH = "dataset"
MODEL_FILE = "trainer.yml"
DATABASE_FILE = "database.csv"

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs("templates", exist_ok=True)

# ── DATABASE SETUP ──
COLUMNS = ["RegisterNo", "Name", "Subject", "Date", "Time", "Status", "Period"]

def load_db():
    if not os.path.exists(DATABASE_FILE):
        pd.DataFrame(columns=COLUMNS).to_csv(DATABASE_FILE, index=False)
    return pd.read_csv(DATABASE_FILE, dtype=str).fillna("")

def save_db(df):
    df.to_csv(DATABASE_FILE, index=False)

# ── FACE PREPROCESS ──
def preprocess_face(gray):
    face = cv2.equalizeHist(gray)
    return cv2.resize(face, (200, 200))

def get_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ── SAVE ATTENDANCE ──
def save_attendance(reg, name, subject, date, time_str, period):
    df = load_db()

    already = (
        (df["RegisterNo"] == reg) &
        (df["Date"] == date) &
        (df["Period"] == period) &
        (df["Subject"] == subject) &
        (df["Status"] == "Present")
    ).any()

    if not already:
        df.loc[len(df)] = [reg, name, subject, date, time_str, "Present", period]
        save_db(df)

# ── HOME ──
@app.route("/")
def home():
    return send_file("templates/index.html")

# ── STUDENTS ──
@app.route("/students")
def students():
    df = load_db()
    s = df[df["Status"] == "Registered"][["RegisterNo", "Name"]].drop_duplicates()
    return jsonify(s.to_dict(orient="records"))

# ── CAPTURE ──
@app.route("/capture", methods=["POST"])
def capture():
    try:
        data = request.get_json()
        reg = str(data.get("reg")).strip()
        name = str(data.get("name")).strip()

        if not reg or not name:
            return jsonify({"message": "Enter Register No and Name"}), 400

        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        detector = get_detector()

        count = 0

        while True:
            ret, img = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                face = preprocess_face(gray[y:y+h, x:x+w])
                count += 1
                cv2.imwrite(f"{DATASET_PATH}/User.{reg}.{count}.jpg", face)

                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            cv2.imshow("Capture", img)

            if cv2.waitKey(1) == 27 or count >= 40:
                break

        cam.release()
        cv2.destroyAllWindows()

        if count == 0:
            return jsonify({"message": "No face detected"}), 400

        df = load_db()
        if df[df["RegisterNo"] == reg].empty:
            df.loc[len(df)] = [reg, name, "", "", "", "Registered", ""]
            save_db(df)

        return jsonify({"message": f"{name} registered successfully ({count} images)"})

    except Exception as e:
        print(e)
        return jsonify({"message": "Capture failed"}), 500

# ── TRAIN ──
@app.route("/train")
def train():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(1,8,8,8)

        faces, ids = [], []

        for file in os.listdir(DATASET_PATH):
            if file.endswith(".jpg"):
                img = cv2.imread(os.path.join(DATASET_PATH, file), 0)
                img = cv2.resize(img, (200,200))
                face_id = int(file.split(".")[1])

                faces.append(img)
                ids.append(face_id)

        if not faces:
            return jsonify({"message": "No data to train"}), 400

        recognizer.train(faces, np.array(ids))
        recognizer.save(MODEL_FILE)

        return jsonify({"message": "Model trained successfully"})

    except Exception as e:
        print(e)
        return jsonify({"message": "Training failed"}), 500

# ── ATTENDANCE ──
@app.route("/attendance")
def attendance():
    try:
        subject = request.args.get("subject", "").strip()
        if not subject:
            return jsonify({"message": "Enter subject"}), 400

        if not os.path.exists(MODEL_FILE):
            return jsonify({"message": "Train model first"}), 400

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_FILE)

        detector = get_detector()
        df = load_db()

        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        last_marked = {}
        COOLDOWN = 30

        while True:
            ret, img = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray,1.2,5)

            for (x,y,w,h) in faces:
                face = preprocess_face(gray[y:y+h,x:x+w])
                face_id, conf = recognizer.predict(face)

                label = "Unknown"
                color = (0,0,255)

                if conf < 90:
                    reg = str(face_id)
                    row = df[(df["RegisterNo"]==reg) & (df["Status"]=="Registered")]

                    if not row.empty:
                        name = row["Name"].values[0]
                        label = name
                        color = (0,255,0)

                        now = datetime.now()

                        if reg in last_marked:
                            if (now - last_marked[reg]).total_seconds() < COOLDOWN:
                                continue

                        last_marked[reg] = now

                        save_attendance(
                            reg, name, subject,
                            now.strftime("%Y-%m-%d"),
                            now.strftime("%H:%M:%S"),
                            "Period"
                        )

                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,label,(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

            cv2.imshow("Attendance", img)

            if cv2.waitKey(1)==27:
                break

        cam.release()
        cv2.destroyAllWindows()

        return jsonify({"message": "Attendance completed"})

    except Exception as e:
        print(e)
        return jsonify({"message": "Attendance failed"}), 500

# ── CLEAR ──
@app.route("/clear_attendance", methods=["POST"])
def clear_attendance():
    try:
        data = request.get_json()
        mode = data.get("mode")

        df = load_db()

        if mode == "all":
            df = df[df["Status"] != "Present"]

        elif mode == "day":
            date = data.get("date")
            df = df[~((df["Status"]=="Present") & (df["Date"]==date))]

        elif mode == "period":
            date = data.get("date")
            period = data.get("period")
            df = df[~((df["Status"]=="Present") & (df["Date"]==date) & (df["Period"]==period))]

        save_db(df)
        return jsonify({"message": "Attendance cleared"})

    except Exception as e:
        print(e)
        return jsonify({"message": "Clear failed"}), 500

# ── DELETE STUDENT ──
@app.route("/delete_student", methods=["POST"])
def delete_student():
    try:
        data = request.get_json()
        mode = data.get("mode")
        df = load_db()

        if mode == "one":
            reg = data.get("reg")
            df = df[df["RegisterNo"] != reg]
            save_db(df)

            for f in os.listdir(DATASET_PATH):
                if f.startswith(f"User.{reg}."):
                    os.remove(os.path.join(DATASET_PATH, f))

            return jsonify({"message": "Student deleted"})

        elif mode == "all":
            save_db(pd.DataFrame(columns=COLUMNS))

            for f in os.listdir(DATASET_PATH):
                os.remove(os.path.join(DATASET_PATH, f))

            if os.path.exists(MODEL_FILE):
                os.remove(MODEL_FILE)

            return jsonify({"message": "All students deleted"})

    except Exception as e:
        print(e)
        return jsonify({"message": "Delete failed"}), 500

# ── VIEW ──
@app.route("/view_all")
def view_all():
    return jsonify(load_db().to_dict(orient="records"))

@app.route("/view_filtered")
def view_filtered():
    df = load_db()
    df = df[df["Status"] == "Present"]

    date = request.args.get("date")
    subject = request.args.get("subject")
    period = request.args.get("period")

    if date:
        df = df[df["Date"] == date]
    if subject:
        df = df[df["Subject"].str.upper() == subject.upper()]
    if period:
        df = df[df["Period"] == period]

    return jsonify(df.to_dict(orient="records"))

# ── RUN ──
if __name__ == "__main__":
    print("✅ SYSTEM RUNNING → http://127.0.0.1:5000")
    app.run(debug=True)
