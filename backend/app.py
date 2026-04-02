from flask import Flask, request, jsonify, render_template
import pickle, sqlite3
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        name TEXT,
        address TEXT,
        glucose REAL,
        bmi REAL,
        age INTEGER,
        prediction TEXT,
        risk REAL,
        level TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

def generate_reasons(data):
    reasons = []

    if data['Glucose'] > 200:
        reasons.append(f"Extremely high glucose ({data['Glucose']}) indicates severe diabetes risk")
    elif data['Glucose'] > 140:
        reasons.append(f"High glucose level ({data['Glucose']})")

    if data['BMI'] > 30:
        reasons.append(f"BMI {data['BMI']} indicates obesity")

    if data['Age'] > 45:
        reasons.append(f"Age {data['Age']} increases diabetes risk")

    return reasons

def generate_advice(data, level):
    advice = []

    if level == "High":
        advice = [
            "Consult doctor immediately",
            "Strict sugar control diet",
            "Daily exercise 30 min"
        ]
    elif level == "Medium":
        advice = [
            "Control diet",
            "Walk daily",
            "Monitor sugar"
        ]
    else:
        advice = [
            "Maintain healthy lifestyle",
            "Regular checkup"
        ]

    if data['BMI'] > 30:
        advice.append("Focus on weight loss")

    return advice

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    features = np.array([[
        data['Pregnancies'], data['Glucose'], data['BloodPressure'],
        data['SkinThickness'], data['Insulin'], data['BMI'],
        data['DiabetesPedigreeFunction'], data['Age']
    ]])

    features = scaler.transform(features)

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    risk = round(prob * 100, 2)

    if data['Glucose'] > 200:
        risk += 20
    if data['BMI'] > 35:
        risk += 10
    if data['Age'] > 60:
        risk += 5

    risk = min(risk, 95)
    

    result = "Diabetic" if pred==1 else "Not Diabetic"

    level = "High" if risk>70 else "Medium" if risk>40 else "Low"

    reasons = generate_reasons(data)
    advice = generate_advice(data, level)

    
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO patients 
    (patient_id,name,address,glucose,bmi,age,prediction,risk,level)
    VALUES (?,?,?,?,?,?,?,?,?)
    """,(
        data['PatientID'], data['Name'], data['Address'],
        data['Glucose'], data['BMI'], data['Age'],
        result, risk, level
    ))

    conn.commit()
    conn.close()

    return jsonify({
        "prediction": result,
        "risk": risk,
        "level": level,
        "reasons": reasons,
        "advice": advice
    })

@app.route('/search')
def search():
    name = request.args.get('name', '')

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM patients WHERE name LIKE ?", ('%'+name+'%',))
    rows = cursor.fetchall()

    conn.close()
    return jsonify(rows)


@app.route('/patients')
def patients():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM patients 
        WHERE id IN (
        SELECT MAX(id) 
        FROM patients 
        GROUP BY patient_id
        )
        """)
    rows = cursor.fetchall()
    conn.close()
    return jsonify(rows)

@app.route('/history/<patient_id>')
def history(patient_id):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT risk FROM patients WHERE patient_id=?", (patient_id,))
    rows = cursor.fetchall()

    conn.close()

    return jsonify([float(r[0]) for r in rows])

if __name__ == "__main__":
    app.run(debug=True)
