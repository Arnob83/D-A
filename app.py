import pickle
import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd
import requests

# URL to the raw xgb_model_new.pkl file in your GitHub repository
url = "https://raw.githubusercontent.com/Arnob83/D-A/main/xgb_model_new.pkl"

# Download the xgb_model_new.pkl file and save it locally
response = requests.get(url)
with open("xgb_model_new.pkl", "wb") as file:
    file.write(response.content)

# Load the trained model
with open("xgb_model_new.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)

@st.cache_data
def prediction(Education_1, ApplicantIncome, CoapplicantIncome, Credit_History, Loan_Amount_Term):
    # Convert user input
    Education_1 = 0 if Education_1 == "Graduate" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1

    # Create input data in the expected order
    input_data = pd.DataFrame(
        [[Education_1, ApplicantIncome, CoapplicantIncome, Credit_History, Loan_Amount_Term]],
        columns=["Education_1", "ApplicantIncome", "CoapplicantIncome", "Credit_History", "Loan_Amount_Term"]
    )

    # Ensure column order matches the classifier’s expectations
    input_data = input_data[classifier.feature_names_in_]

    # Model prediction (0 = Rejected, 1 = Approved)
    prediction = classifier.predict(input_data)
    pred_label = 'Approved' if prediction[0] == 1 else 'Rejected'
    return pred_label, input_data

def explain_with_most_influential_feature(input_data, final_result):
    """
    Analyze features and return the most influential feature contributing to the result,
    along with a bar chart for SHAP values.
    """
    # Initialize SHAP explainer
    explainer = shap.Explainer(classifier)
    shap_values = explainer(input_data)

    # Extract SHAP values for the input data
    shap_values_for_input = shap_values.values[0]  # SHAP values for the first row of input_data

    # Prepare feature importance data
    feature_names = input_data.columns
    contributions = dict(zip(feature_names, shap_values_for_input))

    # Identify the most influential feature
    most_influential_feature = max(contributions, key=lambda k: abs(contributions[k]))
    impact_value = contributions[most_influential_feature]

    explanation_text = (
        f"**Most Influential Feature:** {most_influential_feature}\n\n"
        f"- This feature contributed **{'positively' if impact_value > 0 else 'negatively'}** "
        f"to the result ({final_result}) with a SHAP value of **{impact_value:.2f}**."
    )

    # Create bar chart for SHAP values
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, shap_values_for_input, color="skyblue")
    plt.xlabel("SHAP Value (Impact on Prediction)")
    plt.ylabel("Features")
    plt.title("Feature Contributions to Prediction")
    plt.tight_layout()

    return explanation_text, plt

def main():
    # Front-end elements
    st.markdown(
        """
        <div style="background-color:Yellow;padding:13px">
        <h1 style="color:black;text-align:center;">Loan Prediction ML App</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # User inputs
    Education_1 = st.selectbox('Education', ("Under_Graduate", "Graduate"))
    ApplicantIncome = st.number_input("Applicant's Monthly Income", min_value=0.0)
    CoapplicantIncome = st.number_input("Co-applicant's Monthly Income", min_value=0.0)
    Credit_History = st.selectbox("Credit History", ("Unclear Debts", "Clear Debts"))
    Loan_Amount_Term = st.number_input("Loan Term (in months)", min_value=0.0)

    # Prediction
    if st.button("Predict"):
        result, input_data = prediction(
            Education_1,
            ApplicantIncome,
            CoapplicantIncome,
            Credit_History,
            Loan_Amount_Term
        )

        # Show result in color
        if result == "Approved":
            st.success(f'Your loan is {result}', icon="✅")
        else:
            st.error(f'Your loan is {result}', icon="❌")

        # Explanation: Most Influential Feature and SHAP Bar Plot
        st.header("Most Influential Feature and SHAP Explanation")
        explanation_text, bar_chart = explain_with_most_influential_feature(input_data, final_result=result)
        st.write(explanation_text)
        st.pyplot(bar_chart)

if __name__ == '__main__':
    main()
