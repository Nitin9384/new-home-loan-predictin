from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle


app=Flask(__name__)

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/result',methods=['POST'])
def predict():
    # Getting the data from the form
    loan_amnt=request.form['loan_amnt']
    term=request.form['term']
    int_rate=request.form['int_rate']
    emp_length=request.form['emp_length']
    home_ownership=request.form['home_ownership']
    annual_inc=request.form['annual_inc']
    purpose=request.form['purpose']
    dti=request.form['dti']
    delinq_2yrs=request.form['delinq_2yrs']
    revol_util=request.form['revol_util']
    total_acc=request.form['total_acc']
    longest_credit_length=request.form['longest_credit_length']
    verification_status=request.form['verification_status']
    #  creating a json object to hold the data from the form
    input_data=[{
    'loan_amnt':loan_amnt,
    'term':term,
    'int_rate':int_rate,
    'emp_length':emp_length,
    'home_ownership':home_ownership,
    'annual_inc':annual_inc,
    'purpose':purpose,
    'dti':dti,
    'delinq_2yrs':delinq_2yrs,
    'revol_util':revol_util,
    'total_acc':total_acc,
    'longest_credit_length':longest_credit_length,
    'verification_status':verification_status}]


    dataset=pd.DataFrame(input_data)

    dataset=dataset.rename(columns={
                'loan_amnt':'loan_amnt',
                'term':'term',
                'int_rate':'int_rate',
                'emp_length':'emp_length',
                'home_ownership':'home_ownership',
                'annual_inc':'annual_inc',
                'purpose':'purpose',
                'dti':'dti',
                'delinq_2yrs':'delinq_2yrs',
                'revol_util':'revol_util',
                'total_acc':'total_acc',
                'longest_credit_length':'longest_credit_length',
                'verification_status':'verification_status'})

    dataset[['loan_amnt','int_rate','emp_length','annual_inc', 'dti', 'delinq_2yrs', 'revol_util', 'total_acc','longest_credit_length']] = dataset[['loan_amnt','int_rate','emp_length','annual_inc', 'dti', 'delinq_2yrs', 'revol_util', 'total_acc','longest_credit_length']]

    dataset[['term','home_ownership','purpose','verification_status']]=dataset[['term','home_ownership','purpose','verification_status']].astype('object')

    dataset = dataset[['loan_amnt','term','int_rate','emp_length',
   'annual_inc', 'dti', 'delinq_2yrs', 'revol_util', 'total_acc','longest_credit_length','home_ownership','purpose','verification_status']]
    model = pickle.load(open('Random_Forest.pkl', 'rb'))
    classifier=model.predict(dataset)
    predictions = [item for sublist in RF_cls for item in sublist]
    colors = ['#1f77b4','#ff7f0e']
    loan_status = ['Approved','Not Approved']
    return render_template('index.html',prediction_text=f'Your Loan is {loan_status}')

    p = figure(x_range=loan_status, plot_height=500,
               toolbar_location=None, title="Loan Status", plot_width=800)
    p.vbar(x='loan_status', top='predictions', width=0.4, source=source, legend="loan_status",
           line_color='black', fill_color=factor_cmap('loan_status', palette=colors, factors=loan_status))


    p.xgrid.grid_line_color = None
    p.y_range.start = 0.1
    p.y_range.end = 0.9
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    p.xaxis.axis_label = 'Loan Status'
    p.yaxis.axis_label = ' Predicted Probabilities'
    script, div = components(p)
    return render_template('result.html',script=script,div=div)




if __name__=="__main__":
    app.run(debug=True)