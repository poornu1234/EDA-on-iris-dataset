import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os

# Title of the Streamlit app
st.title("Iris Dataset Exploratory Data Analysis (EDA)")
st.write("This app allows you to explore the famous Iris dataset with interactive visualizations.")

# Step 1: Upload the ZIP file
uploaded_file = st.file_uploader("Upload your Iris dataset ZIP file", type=['zip'])

if uploaded_file is not None:
    # Step 2: Extract the ZIP file
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall('extracted_data')

    # Step 3: Load the CSV file
    csv_file_path = 'extracted_data/iris.csv'
    if os.path.exists(csv_file_path):
        data = pd.read_csv(csv_file_path)
        st.write("### Preview of the dataset:")
        st.dataframe(data.head())

        # Show column names for reference
        st.write("### The columns in this dataset are:")
        st.write(data.columns.tolist())

        # Step 4: Display basic statistics
        st.write("### Basic Statistics")
        st.write(data.describe())

        # Step 5: Select columns for analysis
        st.write("### Select columns for visualization:")
        columns = data.columns.tolist()
        selected_x = st.selectbox("Select X-axis feature", columns)
        selected_y = st.selectbox("Select Y-axis feature", columns)

        # Step 6: Scatter plot
        st.write("### Scatter Plot")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=selected_x, y=selected_y, hue='Species', ax=ax)
        plt.title(f'{selected_x} vs {selected_y}')
        st.pyplot(fig)

        # Step 7: Pairplot
        st.write("### Pairplot of Features")
        pairplot_fig = sns.pairplot(data, hue='Species')
        st.pyplot(pairplot_fig)

        # Step 8: Correlation heatmap (EXCLUDE non-numeric columns)
        st.write("### Correlation Heatmap")
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        fig, ax = plt.subplots()
        sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
        plt.title('Correlation Heatmap')
        st.pyplot(fig)

        # Step 9: Distribution of each feature
        st.write("### Distribution of Features")
        selected_feature = st.selectbox("Select a feature for distribution plot", columns[1:-1])  # Exclude 'Id' and 'Species'
        fig, ax = plt.subplots()
        sns.histplot(data[selected_feature], kde=True, ax=ax)
        plt.title(f'Distribution of {selected_feature}')
        st.pyplot(fig)

        # Step 10: Boxplot for each feature
        st.write("### Boxplot of Features")
        selected_feature_box = st.selectbox("Select a feature for box plot", columns[1:-1])  # Exclude 'Id' and 'Species'
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x='Species', y=selected_feature_box, ax=ax)
        plt.title(f'Boxplot of {selected_feature_box} by Species')
        st.pyplot(fig)
    else:
        st.write("The extracted data does not contain 'iris.csv'. Please check the contents of your ZIP file.")
else:
    st.write("Please upload a ZIP file to proceed.")
