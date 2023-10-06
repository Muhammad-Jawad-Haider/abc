import pandas as pd
from flask import Flask, render_template, request, jsonify
from controller import Controller
from patient import Patient

app = Flask(__name__)

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('Datasets/combineddata3.csv')

# Extract unique values from the 'PatientID' and 'Practice' columns
patient_ids = df['PatientID'].unique().tolist()
practices = df['Practice'].unique().tolist()

prediction = ''
leap_prediction = ''
remedy = ''


@app.route('/', methods=['GET', 'POST'])
def index():
    selected_patient_id = ''
    selected_practice = ''
    patient_ids = df['PatientID'].unique().tolist()
    practices = df['Practice'].unique().tolist()

    if request.method == 'POST':
        selected_patient_id = request.form.get('selected_patient_id')
        selected_practice = request.form.get('selected_practice')

        # Handle the form submission, you can perform any action based on the selected options
        print(f"Selected Patient ID: {selected_patient_id}, Selected Practice: {selected_practice}")

        # patient = Patient(selected_patient_id, selected_practice)
        # patient.load_data('data.csv')

        # controller = Controller()
        # controller.load_model('models/model')
        # prediction = controller.predict(patient.data)
        # leap_prediction, remedy = controller.predict_by_leap(patient.data)


    elif request.method == 'GET':

        patient_ids = df['PatientID'].unique().tolist()
        practices = df['Practice'].unique().tolist()

    return render_template('index.html', patient_ids=patient_ids, practices=practices,
                           selected_patient_id=selected_patient_id, selected_practice=selected_practice,
                           prediction=prediction, leap_prediction=leap_prediction, remedy=remedy)


@app.route('/get_patient_options', methods=['POST'])
def get_patient_options():
    selected_practice = request.form.get('selected_practice')
    print('selected practice: ', selected_practice)

    # Filter DataFrame based on selected practice
    filtered_df = df[df['Practice'] == selected_practice]

    # Extract unique patient IDs for the selected practice
    patient_options = filtered_df['PatientID'].unique().tolist()

    return jsonify(patient_options)


if __name__ == '__main__':
    app.run(debug=True)
