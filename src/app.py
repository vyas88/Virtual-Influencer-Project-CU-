import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import plotly.express as px
from model import VirtualInfluencerNet
from data_processing import preprocess_data, load_data

# Page Config
st.set_page_config(page_title="Virtual Influencer Feasibility", layout="wide")

import json

@st.cache_resource
def load_model_and_scaler():
    try:
        scaler = joblib.load('models/scaler.pkl')
        # Load config (features + targets)
        with open('models/config.json', 'r') as f:
            config = json.load(f)
            
        features = config['features']
        targets = config['targets']
            
        # Load model structure
        input_size = len(features)
        output_size = len(targets)
        model = VirtualInfluencerNet(input_size=input_size, output_size=output_size)
        model.load_state_dict(torch.load('models/vi_model.pth'))
        model.eval()
        return model, scaler, config
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None, None

def run_inference(model, scaler, config, df_processed):
    # Standardize
    features = config['features']
    targets = config['targets']
    
    # We strip targets if they exist to get X potential
    # Actually, we just need to ensure we have the feature columns
    X_potential = df_processed
    
    # Reindex to match training features
    # Fill missing with 0, drop extras
    X = X_potential.reindex(columns=features, fill_value=0)
    
    # Handle numeric conversion
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(X_tensor).numpy()
    
    # Return DataFrame with column names
    pred_df = pd.DataFrame(predictions, columns=targets)
    return pred_df

def main():
    st.title("🤖 Virtual Influencer Market Feasibility Analysis")
    st.markdown("Decision support system powered by Neural Networks.")

    # Sidebar
    st.sidebar.header("Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload Survey Data (Excel/CSV)", type=['xlsx', 'csv'])
    
    # Load default if nothing uploaded
    if not uploaded_file:
        st.sidebar.info("Using default dataset: data/survey_data.xlsx")
        filepath = 'data/survey_data.xlsx'
        df = load_data(filepath)
    else:
        if uploaded_file.name.endswith('.csv'):
             df = pd.read_csv(uploaded_file)
        else:
             df = pd.read_excel(uploaded_file)
    
    if df is not None:
        st.subheader("Data Overview")
        st.dataframe(df.head())
        
        # Load Model
        model, scaler, config = load_model_and_scaler()
        features = config['features'] if config else None
        
        if st.button("Run Analysis"):
            if model is None:
                st.error("Model not found. Please train the model first.")
                return

            with st.spinner("Processing Data and Running Predictions..."):
                # Preprocess
                processed_df, target_cols = preprocess_data(df)
                
                # Predict
                pred_df = run_inference(model, scaler, config, processed_df)
                
                # Add to results
                results = df.copy()
                for t in pred_df.columns:
                    results[f'Predicted_{t}'] = pred_df[t].values
                
                # Metrics
                st.subheader("Prediction Results")
                
                # Display metrics for main targets
                # We specifically care about Satisfaction, but let's show all
                cols = st.columns(len(pred_df.columns))
                
                satisfaction_col = 'vi_satisfaction'
                avg_satisfaction = 0.0
                
                for i, target in enumerate(pred_df.columns):
                    avg_val = pred_df[target].mean()
                    cols[i].metric(f"Avg {target.replace('vi_', '').title()}", f"{avg_val:.2f}")
                    
                    if target == satisfaction_col:
                        avg_satisfaction = avg_val
                        
                # Recommendation Logic based on Satisfaction
                rec_col = st.columns(1)[0]
                if avg_satisfaction >= 3.5:
                    rec = "INVEST / SCALE UP"
                    color = "green"
                elif avg_satisfaction >= 3.0:
                    rec = "PILOT / TEST"
                    color = "orange"
                else:
                    rec = "AVOID / RE-STRATEGIZE"
                    color = "red"
                
                rec_col.markdown(f"**Recommendation (Based on Satisfaction)**")
                rec_col.markdown(f":{color}[**{rec}**]")
                if rec == "INVEST / SCALE UP":
                    rec_col.caption("User sentiment is strong. Proceed with full-scale deployment.")
                elif rec == "PILOT / TEST":
                    rec_col.caption("Mixed sentiment. A limit pilot study is recommended to mitigate risk.")
                else:
                    rec_col.caption("Negative sentiment. Revisit the virtual influencer design or strategy.")

                # Visualizations
                st.subheader("Visualizations")
                
                # Histogram
                fig_hist = px.histogram(results, x="Predicted_vi_satisfaction", nbins=20, title="Distribution of Predicted Satisfaction")
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Scatter if actual exists
                if 'Actual_Satisfaction_Numeric' in results.columns:
                    fig_scat = px.scatter(results, x="Actual_Satisfaction_Numeric", y="Predicted_vi_satisfaction", 
                                          title="Actual vs Predicted Satisfaction", trendline="ols")
                    st.plotly_chart(fig_scat, use_container_width=True)

                # Correlation Matrix
                st.subheader("Correlation Analysis")
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr = processed_df[numeric_cols].corr()
                    fig_corr = px.imshow(corr, text_auto=True, title="Feature Correlation Matrix")
                    st.plotly_chart(fig_corr, use_container_width=True)

                # Feature Drivers (Correlation with Target)
                if satisfaction_col in processed_df.columns:
                    st.subheader("Key Drivers of Satisfaction")
                    correlations = processed_df.corrwith(processed_df[satisfaction_col]).iloc[:-1].sort_values(ascending=False)
                    # Use absolute correlation for importance
                    correlations_abs = correlations.abs().sort_values(ascending=False).head(10)
                    top_drivers = correlations[correlations_abs.index]
                    
                    fig_drivers = px.bar(
                        x=top_drivers.values, 
                        y=top_drivers.index, 
                        orientation='h',
                        title="Top Factors Influencing Satisfaction (Correlation)",
                        labels={'x': 'Correlation with Satisfaction', 'y': 'Feature'}
                    )
                    st.plotly_chart(fig_drivers, use_container_width=True)
                
                st.subheader("Detailed Predictions")
                display_cols = [c for c in results.columns if 'Predicted_' in c]
                st.dataframe(results[display_cols + [c for c in results.columns if c not in display_cols]].head(20))

                # --- Simulation Sandbox ---
                st.markdown("---")
                st.header("🎛 Simulation Sandbox")
                st.markdown("Adjust key parameters to see how they impact predicted satisfaction.")
                
                with st.expander("Run What-If Analysis", expanded=True):
                    # Prepare base features from mean of the dataset
                    # We need the full feature set as expected by the model
                    # The features list from load_model_and_scaler() is the ground truth for input order
                    
                    if model is not None and config is not None:
                        features = config['features']
                        # Create a base input dictionary with zeros (or means where possible)
                        sim_input = {f: 0.0 for f in features}
                        
                        # Use means from current processed_df for numeric columns present in features
                        # This gives a "typical user" baseline
                        numeric_means = processed_df.select_dtypes(include=[np.number]).mean()
                        for f in features:
                            if f in numeric_means:
                                sim_input[f] = float(numeric_means[f])
                        
                        col_sim1, col_sim2 = st.columns(2)
                        
                        # Key Drivers to Adjust
                        # Check which exist in features
                        drivers = ['vi_realism', 'vi_trust', 'vi_engagement_freq', 'age', 'social_media_usage']
                        
                        user_inputs = {}
                        
                        with col_sim1:
                            st.subheader("Perception Factors")
                            if 'vi_realism' in features:
                                user_inputs['vi_realism'] = st.slider("Realism (1-5)", 1.0, 5.0, float(sim_input.get('vi_realism', 3.0)), 0.5)
                            # Trust and Engagement are now TARGETS, so we don't input them.
                            # We predict them!

                        with col_sim2:
                            st.subheader("Demographics")
                            if 'age' in features:
                                user_inputs['age'] = st.slider("Age Group (1-5)", 1.0, 5.0, float(sim_input.get('age', 3.0)), 1.0)
                            if 'social_media_usage' in features:
                                user_inputs['social_media_usage'] = st.slider("Social Media Usage (1-4)", 1.0, 4.0, float(sim_input.get('social_media_usage', 2.0)), 1.0)
                                
                        # Update sim_input with user values
                        for k, v in user_inputs.items():
                            sim_input[k] = v
                            
                        # Handle Interaction Term automatically
                        if 'age_x_usage' in features and 'age' in user_inputs and 'social_media_usage' in user_inputs:
                            sim_input['age_x_usage'] = user_inputs['age'] * user_inputs['social_media_usage']
                            st.caption(f"Calculated Interaction (Age * Usage): {sim_input['age_x_usage']:.1f}")

                        # Create DataFrame for inference
                        sim_df = pd.DataFrame([sim_input])
                        
                        if st.button("Simulate Outcome"):
                            # Predict
                            sim_pred_df = run_inference(model, scaler, config, sim_df)
                            
                            st.subheader("Simulated Outcomes")
                            cols = st.columns(len(sim_pred_df.columns))
                            for i, col in enumerate(sim_pred_df.columns):
                                val = sim_pred_df[col].iloc[0]
                                cols[i].metric(col.replace('vi_', '').title(), f"{val:.2f}/5.0")
                            
                            # Satisfaction logic again for alert
                            sim_sat = sim_pred_df.get('vi_satisfaction', pd.Series([0])).iloc[0]
                            
                            if sim_sat >= 3.5:
                                st.success(f"Likely High Satisfaction ({sim_sat:.2f})")
                            elif sim_sat >= 3.0:
                                st.warning(f"Moderate Satisfaction ({sim_sat:.2f})")
                            else:
                                st.error(f"Likely Low Satisfaction ({sim_sat:.2f})")
                
if __name__ == "__main__":
    main()
