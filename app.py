from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import altair as alt
import os

app = Flask(__name__)

@app.route('/')
def dashboard():
    # Load data
    df = pd.read_csv("mental_health_workplace_survey.csv")

    # ----------- GRAPH 1: Altair Bar Chart (Therapy Access by Gender) ----------- #
    selection = alt.selection_point(fields=['HasTherapyAccess'], bind='legend')
    chart1 = alt.Chart(df).mark_bar().encode(
        x=alt.X('Gender', axis=None),
        y=alt.Y('count()', stack='normalize', axis=alt.Axis(format='%', title='Percentage')),
        color=alt.Color('HasTherapyAccess', title='Has Therapy Access'),
        column=alt.Column('Gender', header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
        tooltip=['Gender', 'HasTherapyAccess', 'count()'],
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
    ).properties(title='Percentage of Employees with Therapy Access by Gender').add_params(selection)
    chart_1 = chart1.to_html()

    # ----------- GRAPH 2: Plotly Histogram (Age Distribution) ----------- #
    fig2 = px.histogram(df, x='Age', nbins=30, marginal='box', title='Age Distribution of Respondents')
    chart_2 = fig2.to_html(full_html=False)

    # ----------- GRAPH 3: Plotly Histogram (Treatment by Gender) ----------- #
    fig3 = px.histogram(df, x='Gender', color='HasTherapyAccess', barmode='group',
                        title='Mental Health Treatment Seeking by Gender')
    chart_3 = fig3.to_html(full_html=False)

    # ----------- GRAPH 4: Violin (Burnout by Remote Work) ----------- #
    fig4 = px.violin(df, x='RemoteWork', y='BurnoutLevel', box=True, points="all",
                     title='Burnout Level Distribution by Remote Work Status',
                     color_discrete_sequence=['pink'])
    chart_4 = fig4.to_html(full_html=False)

    # ----------- GRAPH 5: Pie (Mental Health Support Access) ----------- #
    support_counts = df['HasMentalHealthSupport'].value_counts().reset_index()
    support_counts.columns = ['HasMentalHealthSupport', 'Count']
    fig5 = px.pie(support_counts, values='Count', names='HasMentalHealthSupport',
                  title='Distribution of Mental Health Support Access', hole=0.4)
    chart_5 = fig5.to_html(full_html=False)

    # ----------- GRAPH 6: Box Plot (Burnout by Job Role) ----------- #
    fig6 = px.box(df, x='JobRole', y='BurnoutLevel', title='Burnout Level Distribution by Job Role',
                  color_discrete_sequence=['pink'])
    chart_6 = fig6.to_html(full_html=False)

    # ----------- GRAPH 7: Confusion Matrix ----------- #
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix

    # Features to use in model
    features = ['Age', 'Gender', 'YearsAtCompany', 'WorkHoursPerWeek',
                'RemoteWork', 'BurnoutLevel', 'JobSatisfaction', 'StressLevel',
                'ProductivityScore', 'SleepHours', 'PhysicalActivityHrs',
                'CommuteTime', 'HasMentalHealthSupport', 'ManagerSupportScore',
                'HasTherapyAccess', 'MentalHealthDaysOff', 'SalaryRange',
                'WorkLifeBalanceScore', 'TeamSize', 'CareerGrowthScore',
                'BurnoutRisk']

    df_model = df[features].dropna()

    # Encode categorical variables
    label_encoders = {}
    for col in df_model.columns:
        if df_model[col].dtype == 'object':
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])
            label_encoders[col] = le

    X = df_model.drop('BurnoutRisk', axis=1)
    y = df_model['BurnoutRisk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    labels = ['No Treatment', 'Treatment']
    fig7 = px.imshow(cm, text_auto=True, x=labels, y=labels,
                     labels=dict(x="Predicted", y="Actual", color="Count"),
                     title='Confusion Matrix – Predicting Mental Health Treatment')
    chart_7 = fig7.to_html(full_html=False)

    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': model.coef_[0]})
    fig8 = px.bar(feature_importances.sort_values('importance', ascending=False),
                  x='importance', y='feature', orientation='h',
                  title='Feature Importances from Logistic Regression Model')
    chart_8 = fig8.to_html(full_html=False)

    # ----------- RENDER TEMPLATE ----------- #
    return render_template(
        "dashboard.html",
        chart_1=chart_1,
        chart_2=chart_2,
        chart_3=chart_3,
        chart_4=chart_4,
        chart_5=chart_5,
        chart_6=chart_6,
        chart_7=chart_7,
        chart_8=chart_8
    )

if __name__ == '__main__':
    app.run(debug=True)
