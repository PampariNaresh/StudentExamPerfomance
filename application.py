# from flask import Flask,request,render_template
# import numpy as np
# import pandas as pd

# from sklearn.preprocessing import StandardScaler
# from src.pipeline.predict_pipeline import CustomData,PredictPipeline

# application=Flask(__name__)

# app=application


# ## Route for a home page
# @app.route('/')
# def index():
#     return render_template('home.html') 

# @app.route('/upload', methods=['POST'])
# def table():
#     try:
#         csv_file = request.files['csv_file']
#         if not csv_file:
#             return "No file provided"
#         df = pd.read_csv(csv_file)
#         df = pd.DataFrame(df)
#         df = df.head(30)
#         return render_template('table.html', rows=df.iterrows())
#     except Exception as e:
#         return f"Error: {e}"



# @app.route('/prediction')
# def predict():
#     return render_template('predict.html') 


# @app.route('/home',methods=['GET','POST'])
# def predict_datapoint():
#     if request.method=='GET':
#         return render_template('home.html')
#     else:
#         data=CustomData(
#             gender=request.form.get('gender'),
#             race_ethnicity=request.form.get('ethnicity'),
#             parental_level_of_education=request.form.get('parental_level_of_education'),
#             lunch=request.form.get('lunch'),
#             test_preparation_course=request.form.get('test_preparation_course'),
#             reading_score=float(request.form.get('writing_score')),
#             writing_score=float(request.form.get('reading_score'))

#         )
#         pred_df=data.get_data_as_data_frame()
#         print(pred_df)
#         print("Before Prediction")

#         predict_pipeline=PredictPipeline()
#         print("Mid Prediction")
#         results=predict_pipeline.predict(pred_df)
#         print("after Prediction")
#         return render_template('predict.html',results=results[0])
# if __name__=="__main__":
#     app.run(host="0.0.0.0",debug=True)        



import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from io import StringIO
import base64

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html') 
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']

        # If the user does not select a file, the browser will submit an empty part
        if file.filename == '':
            return "No selected file"

        if file:
            # Read the CSV file using Pandas
            global df
            
            df = pd.read_csv(StringIO(file.read().decode('utf-8')))
            
            
            # Render the HTML template with the data
            return render_template('upload.html', data=df.to_html(classes='table table-striped'))

    return render_template('upload.html', data=None)

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        result =0
        print("after Prediction")
        if results[0]>=100:
            results =results[0]
            
        return render_template('home.html',results=results[0] )

@app.route('/analysis', methods=['POST','GET'])
def analysis():
    df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']
    df['average'] = df['total_score']/3
    df.head()
   
    # df=pd.read_csv(r'C:\Users\Naresh\Desktop\StudentExamPerformance\notebook\data\stud.csv')
    Group_data2=df.groupby('race_ethnicity')
    fig1,ax=plt.subplots(1,3,figsize=(20,8)) #palette = 'mako'
    sns.barplot(x=Group_data2['math_score'].mean().index,y=Group_data2['math_score'].mean().values,ax=ax[0])
    ax[0].set_title('Math score',color='#005ce6',size=20)

    for container in ax[0].containers:
        ax[0].bar_label(container,color='black',size=15)
     #palette = 'flare'
    sns.barplot(x=Group_data2['reading_score'].mean().index,y=Group_data2['reading_score'].mean().values,ax=ax[1])
    ax[1].set_title('Reading score',color='#005ce6',size=20)

    for container in ax[1].containers:
        ax[1].bar_label(container,color='black',size=15)
    #palette = 'coolwarm'
    sns.barplot(x=Group_data2['writing_score'].mean().index,y=Group_data2['writing_score'].mean().values,ax=ax[2])
    ax[2].set_title('Writing score',color='#005ce6',size=20)

    for container in ax[2].containers:
        ax[2].bar_label(container,color='black',size=15)
    fig1.tight_layout()

    buffer1 = io.BytesIO()
    fig1.savefig(buffer1, format='png')
    buffer1.seek(0)
    graph_url1 = base64.b64encode(buffer1.read()).decode()
    plt.close(fig1)

    fig2, ax= plt.subplots(1, 2, figsize=(15, 7))
    plt.subplot(121)
    sns.histplot(data=df,x='average',bins=30,kde=True,color='g')
    plt.subplot(122)
    sns.histplot(data=df,x='average',kde=True,hue='gender')
    fig2.tight_layout()
    buffer2 = io.BytesIO()
    fig2.savefig(buffer2, format='png')
    buffer2.seek(0)
    graph_url2 = base64.b64encode(buffer2.read()).decode()
    # Close the Matplotlib figures
    plt.close(fig2)



    fig3, ax = plt.subplots(1, 2, figsize=(15, 7))
    plt.subplot(121)
    sns.histplot(data=df,x='total_score',bins=30,kde=True,color='g')
    plt.subplot(122)
    sns.histplot(data=df,x='total_score',kde=True,hue='gender')
    
    buffer3 = io.BytesIO()
    fig3.savefig(buffer3, format='png')
    buffer3.seek(0)
    graph_url3 = base64.b64encode(buffer3.read()).decode()
    # Close the Matplotlib figures
    plt.close(fig3)

    fig4,ax =plt.subplots(1,3,figsize=(25,6))
    plt.subplot(141)
    sns.histplot(data=df,x='average',kde=True,hue='lunch')
    plt.subplot(142)
    sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='lunch')
    plt.subplot(143)
    sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='lunch')
    buffer4 = io.BytesIO()
    fig4.savefig(buffer4, format='png')
    buffer4.seek(0)
    graph_url4 = base64.b64encode(buffer4.read()).decode()
    # Close the Matplotlib figures
    plt.close(fig4)


    fig5,ax =plt.subplots(1,3,figsize=(25,6))
    plt.subplot(141)
    ax =sns.histplot(data=df,x='average',kde=True,hue='parental_level_of_education')
    plt.subplot(142)
    ax = sns.histplot(data=df[df.gender == 'male'], x='average',
                  kde=True, hue='parental_level_of_education')
    plt.subplot(143)
    ax = sns.histplot(data=df[df.gender == 'female'], x='average',
                  kde=True, hue='parental_level_of_education')
    buffer5 = io.BytesIO()
    fig5.savefig(buffer5, format='png')
    buffer5.seek(0)
    graph_url5 = base64.b64encode(buffer5.read()).decode()
    # Close the Matplotlib figures
    plt.close(fig5)

    fig6,ax =plt.subplots(1,3,figsize=(25,6))
    plt.subplot(141)
    ax =sns.histplot(data=df,x='average',kde=True,hue='race_ethnicity')
    plt.subplot(142)
    ax =sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='race_ethnicity')
    plt.subplot(143)
    ax =sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='race_ethnicity')
    
    buffer6 = io.BytesIO()
    fig6.savefig(buffer6, format='png')
    buffer6.seek(0)
    graph_url6 = base64.b64encode(buffer6.read()).decode()
    # Close the Matplotlib figures
    plt.close(fig6)


    fig7,ax=plt.subplots(1,2,figsize=(20,10))
    sns.countplot(x=df['gender'],data=df,palette ='bright',ax=ax[0],saturation=0.95)
    for container in ax[0].containers:
        ax[0].bar_label(container,color='black',size=20)
    
    plt.pie(x=df['gender'].value_counts(),labels=['Male','Female'],explode=[0,0.1],autopct='%1.1f%%',shadow=True,colors=['#ff4d4d','#ff8000'])
    buffer7 = io.BytesIO()
    fig7.savefig(buffer7, format='png')
    buffer7.seek(0)
    graph_url7 = base64.b64encode(buffer7.read()).decode()
    # Close the Matplotlib figures
    plt.close(fig7)
    plt.clf()
   




    return render_template('analysis.html', graph_url1=f"data:image/png;base64,{graph_url1}", graph_url2=f"data:image/png;base64,{graph_url2}",graph_url3=f"data:image/png;base64,{graph_url3}",graph_url4=f"data:image/png;base64,{graph_url4}",graph_url5=f"data:image/png;base64,{graph_url5}",graph_url6=f"data:image/png;base64,{graph_url6}",graph_url7=f"data:image/png;base64,{graph_url7}")
    # return render_template('analysis.html', graph_url1=f"data:image/png;base64,{graph_url1}")


@app.route('/abstract')
def abstract():
    return render_template('abstract.html')  # Create an 'abstract.html' template

if __name__=="__main__":
    #app.run(host="0.0.0.0",debug=True) 
    app.run()        

