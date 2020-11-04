import re
import dash
import joblib
import numpy as np
import xgboost as xgb
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import base64



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, title='InFraudmatics',
                update_title='InFraudmatics App',
                external_stylesheets=external_stylesheets)

server = app.server

df = pd.read_csv("phys_new.csv")
model = joblib.load("final.pkl", mmap_mode="r")
morb_df = pd.read_csv("disease_data.csv")

value = {
    "AmtReimbursed": 60,
    "DeductibleAmt": 0,
    "Is_inpatient": 0,
    "Gender": 1,
    "RenalDisease": 0,
    "State": 25,
    "County": 40,
    "InpatientAnnualDeductibleAmt": 1,
    "OutpatientAnnualDeductibleAmt": 0,
    "Age": 0,
    "DiagnosisCode_Num": 1,
    "ProcedureCode_Num": 0,
    "ChronicDisease_Num": 4,
    "PhysiciansNum": 1,
    "AdmitDiagnosInDiagnos": 0,
    "FullYearPlanA": 1,
    "FullYearPlanB": 1,
    "Is_Dead": 0
    }

df['Fraud'] = df['Fraud'].astype(str)

physicians = [{"label": str(k), "value": str(k)} for k in df["AttendingPhysician"]]

def score(num):
  return str((1 + float(num)) / 2)

def card_description():
    return html.Div(
        id="description-card",
        children=[
            html.H5("InFraudmatics", style={"color": "#3412d4", "fontWeight": "bold", "fontSize": "28px"}),
            html.H3("Welcome to the Dashboard", style={"fontWeight": "bold"}),
            html.Div(
                id="Purpose",
                children="Explore potential fraud and cut business loss."
            )
        ]
    )

image_filename = 'xgplot.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div(children=[
    html.Div(
      id="left-column",
      className="column",
      style={"height": "100%", "width": "30%", "position": "fixed",
           "zIndex": 1, "top": 0, "left": 0, "paddingLeft": "2%", "paddingRight": "2%"},
      children=[card_description()]
      ),
    html.Div(
      id="right-column",
      style={"paddingLeft": "30%"},
      children=[
        dcc.Tabs(id="tabs", value="tab-1",
          children=[
            # Morbidity Analysis
            dcc.Tab(label="Morbidity Analysis", value="tab-1",
                children=[
                  # TODO: Continue work from here...
                  dcc.Graph(figure=px.bar(morb_df, x="disease", y="count", color="type"))
                  ]),

            # Physician Analysis
            dcc.Tab(label="Physician History", value="tab-2",
                    children=[
                      html.Div(id="info",
                               children=[
                                    html.H3("Has the physician commited fraud in the past?"),
                                    dcc.Dropdown(id="physician-dropdown",
                                      style={"width": "60%"}),
                                    html.H2("KPIs"),
                                    html.Div(
                                        [
                                            html.Div(
                                              [html.H6(id="potential-fraud", children="12", style={"font-size": "48px", "font-weight": "bold"}), html.P("No. of Potential Fraud")],
                                                id="wells",
                                                className="mini_container",
                                                style={"padding": "2%", "filter": "drop-shadow(2px 2px 2px #d4d4d4)", "background-color": "#eeeeee", "margin": "20px", "width": "200px", "textAlign": "center"}
                                            ),
                                            html.Div(
                                                [html.H6(id="claims", children="12", style={"font-size": "48px", "font-weight": "bold"}), html.P("No.of claims")],
                                                id="gas",
                                                className="mini_container",
                                                style={"padding": "2%", "filter": "drop-shadow(2px 2px 2px #d4d4d4)", "background-color": "#eeeeee", "margin": "20px", "width": "200px", "textAlign": "center"}

                                            ),
                                            html.Div(
                                                [html.H6(id="providers", children="12", style={"font-size": "48px", "font-weight": "bold"}), html.P("No. of providers")],
                                                id="oil",
                                                className="mini_container",
                                                style={"padding": "2%", "filter": "drop-shadow(2px 2px 2px #d4d4d4)", "background-color": "#eeeeee", "margin": "20px", "width": "200px", "textAlign": "center"}

                                            )
                                        ],
                                        id="info-container",
                                        className="row container-display",
                                        style={"display": "flex"}
                                    ),
                                  html.Div(
                                    [
                                      dcc.Graph(
                                        id="fraud-instances",
                                        style={"width": "40%"},
                                        figure={
                                          'data':[]
                                          }
                                        ),
                                      dcc.Graph(
                                        id="fraud-instances-2",
                                        style={"width": "40%"},
                                        figure=px.scatter(df, x="AttendingPhysician", y="AmtReimbursed", color="Fraud")
                                        ),
                                    ],
                                    id='fraud-level',
                                    style={'display': 'flex'}
                                    )
                                 ])
                      ]),

            # ML Model
            dcc.Tab(label="Real time predictions", value="tab-3",
                    children=[
                      html.Div(
                        [html.H2("Predictions"),
                         html.Div(
                           [html.Img(src="data:image/png;base64,{}".format(encoded_image.decode()))],
                           style={'float': "right", "paddingRight": "5%", "paddingTop": "5%"}
                           ),
                         # AmtReimbursed
                         html.Div([html.Label("Amount Reimbursed: ", style={
                           "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                         html.H1(" "),
                         dcc.Input(
                           id="AmtReimbursed",
                           type="text",
                           placeholder="",
                           value=""
                           )]
                           , id="amt-input",
                           style={"display": "flex", "marginBottom": "15px"}
                         ),
                         # DeductibleAmt
                         html.Div([html.Label("Is Deductible Amount zero? ", style={
                           "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                         html.H1(" "),
                         dcc.Dropdown(
                           id="DeductibleAmt",
                           options=[{"label": "True" , "value": 0},
                                    {"label": "False", "value": 1}],
                           value="",
                           style={"width": "60%"}
                           )]
                           , id="ded-input",
                           style={"display": "flex", "marginBottom": "15px"}
                         ),
                         # Is_inpatient
                         html.Div([html.Label("Is Inpatient? ", style={
                           "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                         html.H1(" "),
                         dcc.Dropdown(
                           id="Is_inpatient",
                           options=[{"label": "True" , "value": 1},
                                    {"label": "False", "value": 0}],
                           value="",
                           style={"width": "60%"}
                           )]
                           , id="inp-input",
                           style={"display": "flex", "marginBottom": "15px"}
                         ),
                         # Gender
                         html.Div([html.Label("Gender: ", style={
                           "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                         html.H1(" "),
                         dcc.Dropdown(
                           id="Gender",
                           options=[{"label": "Male" , "value": 1},
                                    {"label": "Female", "value": 2}],
                           value="",
                           style={"width": "60%"}
                           )]
                           , id="gen-input",
                           style={"display": "flex", "marginBottom": "15px"}
                         ),
                         # RenalDisease
                         html.Div([html.Label("Is Renal disease? ", style={
                           "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                         html.H1(" "),
                         dcc.Dropdown(
                           id="RenalDisease",
                           options=[{"label": "True" , "value": 1},
                                    {"label": "False", "value": 0}],
                           value="",
                           style={"width": "60%"}
                           )]
                           , id="ren-input",
                           style={"display": "flex", "marginBottom": "15px"}
                         ),
                         # State
                         html.Div([html.Label("State: ", style={
                           "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                         html.H1(" "),
                         dcc.Input(
                           id="State",
                           type="text",
                           placeholder="",
                           value=""
                           )]
                           , id="state-input",
                           style={"display": "flex", "marginBottom": "15px"}
                         ),
                         # County
                         html.Div([html.Label("County: ", style={
                           "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                         html.H1(" "),
                         dcc.Input(
                           id="County",
                           type="text",
                           placeholder="",
                           value=""
                           )]
                           , id="county-input",
                           style={"display": "flex", "marginBottom": "15px"}
                         ),
# "InpatientAnnualDeductibleAmt"
# "OutpatientAnnualDeductibleAmt"
                          # InpatientAnnualDeductibleAmt
                          html.Div([html.Label("Is Inpatient Annual Deductible Amount zero? ", style={
                            "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                          html.H1(" "),
                          dcc.Dropdown(
                            id="InpatientAnnualDeductibleAmt",
                            options=[{"label": "True" , "value": 1},
                                     {"label": "False", "value": 0}],
                            value="",
                            style={"width": "60%"}
                            )]
                            , id="inpded-input",
                            style={"display": "flex", "marginBottom": "15px"}
                          ),
                          # OutpatientAnnualDeductibleAmt
                          html.Div([html.Label("Is Outpatient Annual Deductible Amount zero? ", style={
                            "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                          html.H1(" "),
                          dcc.Dropdown(
                            id="OutpatientAnnualDeductibleAmt",
                            options=[{"label": "True" , "value": 1},
                                     {"label": "False", "value": 0}],
                            value="",
                            style={"width": "60%"}
                            )]
                            , id="outpded-input",
                            style={"display": "flex", "marginBottom": "15px"}
                          ),
                         # Age
                         html.Div([html.Label("Age: ", style={
                           "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                         html.H1(" "),
                         dcc.Input(
                           id="Age",
                           type="text",
                           placeholder="",
                           value=""
                           )]
                           , id="age-input",
                           style={"display": "flex", "marginBottom": "15px"}
                         ),
                         # DiagnosisCode_Num
                         html.Div([html.Label("Number of Diagnosis code: ", style={
                           "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                         html.H1(" "),
                         dcc.Input(
                           id="DiagnosisCode_Num",
                           type="text",
                           placeholder="",
                           value=""
                           )]
                           , id="diag-input",
                           style={"display": "flex", "marginBottom": "15px"}
                         ),
                         # ProcedureCode_Num
                         html.Div([html.Label("Number of Procedure code: ", style={
                           "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                         html.H1(" "),
                         dcc.Input(
                           id="ProcedureCode_Num",
                           type="text",
                           placeholder="",
                           value=""
                           )]
                           , id="proc-input",
                           style={"display": "flex", "marginBottom": "15px"}
                         ),
                         # ChronicDisease_Num
                         html.Div([html.Label("Number of Chronic diseases: ", style={
                           "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                         html.H1(" "),
                         dcc.Input(
                           id="ChronicDisease_Num",
                           type="text",
                           placeholder="",
                           value=""
                           )]
                           , id="chron-input",
                           style={"display": "flex", "marginBottom": "15px"}
                         ),
                         # Number of physicians
                         html.Div([html.Label("Number of Physicians: ", style={
                           "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                         html.H1(" "),
                         dcc.Input(
                           id="PhysiciansNum",
                           type="text",
                           placeholder="",
                           value=""
                           )]
                           , id="phy-input",
                           style={"display": "flex", "marginBottom": "15px"}
                         ),
                         # AdmitDiagnosInDiagnos
                          html.Div([html.Label("Is Admit Diagnosis Code in Diagnosis Code? ", style={
                            "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                          html.H1(" "),
                          dcc.Dropdown(
                            id="AdmitDiagnosInDiagnos",
                            options=[{"label": "True" , "value": 1},
                                     {"label": "False", "value": 0}],
                            value="",
                            style={"width": "60%"}
                            )]
                            , id="admit-input",
                            style={"display": "flex", "marginBottom": "15px"}
                          ),
                          # FullYearPlanA
                          html.Div([html.Label("Has Full Year Plan A? ", style={
                            "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                          html.H1(" "),
                          dcc.Dropdown(
                            id="FullYearPlanA",
                            options=[{"label": "True" , "value": 1},
                                     {"label": "False", "value": 0}],
                            value="",
                            style={"width": "60%"}
                            )]
                            , id="fullA-input",
                            style={"display": "flex", "marginBottom": "15px"}
                          ),
                          # FullYearPlanB
                          html.Div([html.Label("Has Full Year Plan B? ", style={
                            "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                          html.H1(" "),
                          dcc.Dropdown(
                            id="FullYearPlanB",
                            options=[{"label": "True" , "value": 1},
                                     {"label": "False", "value": 0}],
                            value="",
                            style={"width": "60%"}
                            )]
                            , id="fullB-input",
                            style={"display": "flex", "marginBottom": "15px"}
                          ),
                          # Is_Dead
                          html.Div([html.Label("Is dead? ", style={
                            "display": "flex", "alignItems": "center", "paddingRight": "2%"}),
                          html.H1(" "),
                          dcc.Dropdown(
                            id="Is_Dead",
                            options=[{"label": "True" , "value": 1},
                                     {"label": "False", "value": 0}],
                            value="",
                            style={"width": "60%"}
                            )]
                            , id="dead-input",
                            style={"display": "flex", "marginBottom": "15px"}
                          ),
                          # Submit
                          html.Div(id="target"),
                          html.Div(id="conc-label", children="The result is "),
                          html.Div(id="result", style={"fontWeight": "bold", "fontSize": "24px"}),
                          html.Div(id="res")
                       ])
                      ])
                ]
              )
        ]
      )
  ])


def check_bool(value, types):
  return value in types


@app.callback(Output('result', 'children'),
              [
Input("AmtReimbursed"                , 'value'),
Input("DeductibleAmt"                , 'value'),
Input("Is_inpatient"                 , 'value'),
Input("Gender"                       , 'value'),
Input("RenalDisease"                 , 'value'),
Input("State"                        , 'value'),
Input("County"                       , 'value'),
Input("InpatientAnnualDeductibleAmt" , 'value'),
Input("OutpatientAnnualDeductibleAmt", 'value'),
Input("Age"                          , 'value'),
Input("DiagnosisCode_Num"            , 'value'),
Input("ProcedureCode_Num"            , 'value'),
Input("ChronicDisease_Num"           , 'value'),
Input("PhysiciansNum"                , 'value'),
Input("AdmitDiagnosInDiagnos"        , 'value'),
Input("FullYearPlanA"                , 'value'),
Input("FullYearPlanB"                , 'value'),
Input("Is_Dead"                      , 'value')])
def save_information(amt_reimbursed, deductible_amt, is_inpatient, gender, renal_disease, state, county, inded, outded, age, diag, proc, chronic, physician, admit, fullA, fullB, is_dead):
  try:
    value['AmtReimbursed'] = int(amt_reimbursed) if check_bool(int(amt_reimbursed), range(0, 100000000)) else 0
    value["DeductibleAmt"] = int(deductible_amt) if check_bool(int(deductible_amt), [0, 1]) else 0
    value["Is_inpatient"] = int(is_inpatient) if check_bool(int(is_inpatient), [0,1]) else 0
    value["Gender"] = int(gender) if check_bool(int(gender), [1,2]) else 1
    value["RenalDisease"] = int(renal_disease) if check_bool(int(renal_disease), [1,0]) else 0
    value["State"] = int(state) if check_bool(int(state), range(0,51)) else 0
    value["County"] = int(county) if check_bool(int(county), range(0,51)) else 0
    value["InpatientAnnualDeductibleAmt"]= int(inded) if check_bool(int(inded), [0,1]) else 0
    value["OutpatientAnnualDeductibleAmt"] = int(outded) if check_bool(int(outded), [0,1]) else 0
    value["Age"] = int(age) if check_bool(int(age), range(0, 200)) else 0
    value["DiagnosisCode_Num"] = int(diag) if check_bool(int(diag), range(0,50)) else 0
    value["ProcedureCode_Num"] = int(proc) if check_bool(int(proc), range(0,50)) else 0
    value["ChronicDisease_Num"] = int(chronic) if check_bool(int(chronic), range(0,50)) else 0
    value["PhysiciansNum"] = int(physician) if check_bool(int(physician), range(0,50)) else 0
    value["AdmitDiagnosInDiagnos"] = int(admit) if check_bool(int(admit), [0,1]) else 0
    value["FullYearPlanA"] = int(fullA) if check_bool(int(fullA), [0,1]) else 0
    value["FullYearPlanB"] = int(fullB) if check_bool(int(fullB), [0,1]) else 0
    value["Is_Dead"] = int(is_dead) if check_bool(int(is_dead), [0,1]) else 0
  except:
    pass

    res = model.predict(xgb.DMatrix(pd.DataFrame({k: [v] for k, v in value.items()})))
    res = score(res[0])
    if float(res) < 0.5:
      result = "Not fraud"
    else:
      result = "Fraud"
    return "{0}".format(result)

@app.callback(
    [dash.dependencies.Output("fraud-instances", "figure"),
     dash.dependencies.Output("potential-fraud", "children"),
     dash.dependencies.Output("claims", "children"),
     dash.dependencies.Output("providers", "children")

     ],
    [dash.dependencies.Input("physician-dropdown", "value")]
    )
def update_physician_graphs(value):
  new_df = df.copy()
  new_df = new_df[new_df["AttendingPhysician"] == value]
  fraud = len(new_df[new_df["Fraud"] == '1'])
  not_fraud = len(new_df[new_df["Fraud"] == '0'])
  return {'data': [{'x': ["Fraud", "Not Fraud"], 'y': [fraud, not_fraud], 'type': "bar"}]}, str(fraud), str(fraud + not_fraud), str(len(new_df["PID"].unique()))

@app.callback(
    dash.dependencies.Output("physician-dropdown", "options"),
    [dash.dependencies.Input("physician-dropdown", "search_value")],
    [dash.dependencies.State("physician-dropdown", "value")],
)
def update_physicians(search_value, value):
  if not search_value:
    raise PreventUpdate
  search_results = [o for o in physicians if search_value in o["label"]][:5]
  return search_results


if __name__ == '__main__':
  app.run_server(debug=True)

