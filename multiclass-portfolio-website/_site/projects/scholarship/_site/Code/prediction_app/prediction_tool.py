# Aid Worker Security Incident Fatality Prediction Tool

import os
import sys
import traceback
from shiny import ui, App, render, reactive
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# load model 
model_path = 'model/fatality_prediction_model.joblib'
threshold_path = 'model/optimal_threshold.joblib'
feature_importance_path = 'model/feature_importance.csv'

model = joblib.load(model_path)
threshold = joblib.load(threshold_path)
feature_importance = pd.read_csv(feature_importance_path)

# define dropdowns
attack_types = ['Unknown', 'Shooting', 'Kidnapping', 'Kidnap-killing', 'Aerial bombardment', 'Landmine ', 'Shelling', 'Body-borne IED', 'Bodily assault', 'Roadside IED', 'Vehicle-born IED', 'Other Explosives', 'Rape/sexual assault ', 'Complex attack']
attack_contexts = ['Unknown', 'Raid', 'Individual attack', 'Combat/Crossfire', 'Ambush', 'Detention', 'Mob violence']
locations = ['Unknown', 'Office/compound', 'Road', 'Home', 'Project site', 'Custody', 'Public location']
countries = ['Cambodia', 'Rwanda', 'Tajikistan', 'Somalia', 'DR Congo', 'Sierra Leone', 'Chechnya', 'Bangladesh', 'South Sudan', 'Ethiopia', 'Angola', 'Afghanistan', 'Congo', 'Burundi', 'Sri Lanka', 'Uganda', 'Sudan', 'Kosovo', 'Yemen', 'Indonesia', 'Liberia', 'Kenya', 'Colombia', 'Mozambique', 'Iraq', 'Georgia', 'Guinea', 'Cameroon', 'Madagascar', 'Fiji', 'Central African Republic', 'Pakistan', 'Occupied Palestinian Territories', 'South Africa', 'Jordan', 'Kashmir', "Cote D'Ivoire", 'Senegal', 'Eritrea', 'Philippines', 'Nigeria', 'Haiti', 'Zimbabwe', 'Nepal', 'Malawi', 'Chad', 'Guyana', 'Guatemala', 'India', 'El Salvador', 'Lesotho', 'Lebanon', 'Vietnam', 'Mali', 'Tanzania', 'Papua New Guinea', 'Thailand', 'Algeria', 'Nicaragua', 'Zambia', 'Mauritania', 'Niger', 'Kyrgyzstan', 'Libyan Arab Jamahiriya', 'Myanmar', 'Honduras', 'Syrian Arab Republic', 'Western Sahara', 'Tunisia', 'Turkey', 'Armenia', 'Argentina', 'Ukraine', 'Benin', 'Israel', 'Burkina Faso', 'Mexico', 'Mauritius', 'Saudi Arabia', 'Azerbaijan', 'Bolivia', 'Ecuador', 'Peru', 'Chile', 'Botswana', 'Uruguay', 'Namibia', 'Guinea-Bissau', 'Swaziland', 'Poland', 'Iran, Islamic Republic of', 'Egypt', 'Dominican Republic', 'Venezuela', 'Jamaica']
actor_types = ['Unknown', 'Non-state armed group: Regional', 'Non-state armed group: National', 'Non-state armed group: Unknown', 'Staff member', 'Non-state armed group: Subnational', 'Unaffiliated', 'State: unknown', 'Police or paramilitary', 'Host State', 'Aid recipient', 'Non-state armed group: Global', 'Criminal', 'Foreign or coalition forces']
motives = ['Unknown', 'Political', 'Incidental', 'Economic', 'Disputed', 'Other']

# Utility functions
def prepare_incident_for_prediction(incident_dict):
    """Prepare incident for prediction."""
    incident_df = pd.DataFrame([incident_dict])
    
    # match model features
    expected_features = model.feature_names_in_
    missing_features = set(expected_features) - set(incident_df.columns)
    
    if missing_features:
        for feature in missing_features:
            incident_df[feature] = 0
    
    incident_df = incident_df.reindex(columns=expected_features, fill_value=0)
    
    return incident_df

def predict_fatality(incident_df, threshold=0.5):
    """Make prediction."""
    prob_fatal = model.predict_proba(incident_df)[0, 1]
    is_fatal = prob_fatal >= threshold
    
    # confidence determination
    def get_confidence(prob):
        if abs(prob - 0.5) > 0.4:
            return 'Very High'
        elif abs(prob - 0.5) > 0.3:
            return 'High'
        elif abs(prob - 0.5) > 0.2:
            return 'Medium'
        elif abs(prob - 0.5) > 0.1:
            return 'Low'
        return 'Very Low'
    
    return {
        'prediction': 'Fatal' if is_fatal else 'Non-Fatal',
        'probability': prob_fatal,
        'confidence': get_confidence(prob_fatal)
    }

# UI Definition
app_ui = ui.page_fluid(
    ui.HTML("""
    <style>
        .risk-bar {
            width: 100%;
            height: 30px;
            background: linear-gradient(to right, 
                green 0%, 
                yellowgreen 25%, 
                yellow 50%, 
                orange 75%, 
                red 100%
            );
            margin-top: 15px;
            position: relative;
        }
        .risk-marker {
            position: absolute;
            width: 4px;
            height: 40px;
            background-color: black;
            top: -5px;
        }
        .risk-labels {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            margin-top: 5px;
        }
    </style>
    """),
    ui.h2("Aid Worker Security Incident Fatality Prediction Tool"),
    ui.div(
        {"style": "max-width: 900px; margin: 0 auto; padding: 20px;"},
        
        ui.row(
            ui.column(6,
                ui.h3("Incident Characteristics"),
                ui.input_select("attack_type", "Attack type:", attack_types),
                ui.input_select("context", "Context:", attack_contexts),
                ui.input_select("location", "Location:", locations),
                ui.input_select("country", "Country:", countries),
                ui.input_select("actor_type", "Actor type:", actor_types),
                ui.input_select("motive", "Motive:", motives),
            ),
            
            ui.column(6,
                ui.h3("Victim Information"),
                ui.input_slider("total_affected", "Total affected:", min=1, max=20, value=3),
                ui.input_slider("org_types_count", "Org. types affected:", min=1, max=6, value=1),
                ui.input_slider("intl_proportion", "Intl. proportion:", min=0, max=1, step=0.1, value=0.3),
                ui.input_slider("male_percent", "% Male:", min=0, max=100, step=5, value=60),
                ui.input_slider("female_percent", "% Female:", min=0, max=100, step=5, value=30),
            )
        ),
        
        ui.row(
            ui.column(12,
                ui.div(
                    {"id": "results", "style": "margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: white;"},
                    ui.output_ui("prediction_result")
                )
            )
        )
    )
)


def server(input, output, session):
    @output
    @render.ui
    def prediction_result():
        # validate male and female results
        if input.male_percent() + input.female_percent() > 100:
            return ui.div("Error: Male and female percentages cannot exceed 100%", style="color: red;")
        
        # incident dictionary
        incident = {
            'Means of attack': input.attack_type(),
            'Attack context': input.context(),
            'Location': input.location(),
            'Country': input.country(),
            'Actor type': input.actor_type(),
            'Motive': input.motive(),
            'Total affected': input.total_affected(),
            'org_types_affected': input.org_types_count(),
            'intl_proportion': input.intl_proportion(),
            'percent_male_affected': input.male_percent(),
            'percent_female_affected': input.female_percent()
        }
        
        # prepare and predict
        incident_df = prepare_incident_for_prediction(incident)
        result = predict_fatality(incident_df, threshold=threshold)
        
        # determine risk bar position 
        risk_position = f"{result['probability'] * 100:.1f}%"
        
        # results
        color = 'red' if result['prediction'] == 'Fatal' else 'green'
        
        return ui.div(
            ui.h4("Prediction Results:"),
            ui.tags.p([
                "Prediction: ", 
                ui.tags.span(result['prediction'], style=f"color:{color}; font-weight:bold")
            ]),
            ui.p(f"Probability of fatality: {result['probability']:.1%}"),
            ui.p(f"Confidence level: {result['confidence']}"),
            ui.HTML(f'''
                <div class="risk-bar">
                    <div class="risk-marker" style="left: {risk_position};"></div>
                </div>
                <div class="risk-labels">
                    <span>Very Low Risk</span>
                    <span>Low Risk</span>
                    <span>Medium Risk</span>
                    <span>High Risk</span>
                    <span>Very High Risk</span>
                </div>
            ''')
        )

# create the app
app = App(app_ui, server)