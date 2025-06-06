from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app = application

#Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictchurn',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            satisfaction_level = request.form.get('satisfaction_level'),
            last_evaluation = request.form.get('last_evaluation'),
            number_project = request.form.get('number_project'),
            average_montly_hours = request.form.get('average_montly_hours'),
            time_spend_company = request.form.get('time_spend_company'),
            Department = request.form.get('Department'),
            salary = request.form.get('salary'),
            Work_accident = request.form.get('Work_accident'),
            promotion_last_5years = request.form.get('promotion_last_5years')
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)
        prediction_label = "Employee will Leave" if prediction[0] == 1 else "Employee will Stay"
        print(prediction)
        return render_template('home.html',prediction=prediction_label)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)

