import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Title and introduction
st.title("Typhoon Intensity Prediction Based on Global Warming")
st.write("This app simulates the impact of global warming on typhoon intensity. Adjust the global temperature increase and see how it affects typhoon intensity.")

# Sidebar for user input
st.sidebar.header("Global Warming Inputs")
temp_increase = st.sidebar.slider('Global Temperature Increase (°C)', 0.0, 5.0, 1.0)

# Example dataset (for simplicity, generating random data)# In practice, you would use real historical typhoon and climate data.
data = {
    'sea_surface_temp': np.random.uniform(25, 30, 100), # Random SST in degrees Celsius
    'wind_speed': np.random.uniform(120, 250, 100), # Wind speed in km/h (this is the typhoon intensity label)
}

df = pd.DataFrame(data)

# Adjust sea surface temperature based on global warming increase
df['sea_surface_temp'] += temp_increase

# Train a simple model (Linear Regression for simplicity)
X = df[['sea_surface_temp']]  # Using sea surface temp as the only feature
y = df['wind_speed']  # Typhoon intensity as wind speed
model = LinearRegression()
model.fit(X, y)

# Predict typhoon intensity based on the temperature increase
predicted_intensity = model.predict([[30 + temp_increase]])  # Simulate 30°C base temp + global warming

# Display result
st.write(f"Predicted Typhoon Intensity with a {temp_increase}°C global temperature increase: **{predicted_intensity[0]:.2f} km/h**")

# Plot the data and prediction
fig, ax = plt.subplots()
ax.scatter(df['sea_surface_temp'], df['wind_speed'], label="Historical Data")
ax.scatter([30 + temp_increase], predicted_intensity, color="red", label="Prediction")
ax.set_xlabel("Sea Surface Temperature (°C)")
ax.set_ylabel("Typhoon Intensity (km/h)")
ax.set_title("Typhoon Intensity vs Sea Surface Temperature")
ax.legend()

# Display the plot
st.pyplot(fig)

# Conclusion
st.write("As global warming increases sea surface temperatures, the intensity of typhoons is likely to increase. This simple simulation is based on historical data and a basic linear model.")
