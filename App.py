import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error


from Data.Lorenz import generate_lorenz_data
from Model.esn import EchoStateNetwork

# Page Configuration 
st.set_page_config(page_title="ESN Chaos Predictor", page_icon="🦋", layout="wide")

st.title("🦋 Chaotic System Forecasting: The Edge of Stability")
st.markdown("""
This interactive dashboard demonstrates the predictive power of an Echo State Network on the Lorenz Attractor. 
Based on the thesis *"Is the edge of stability Good for Reservoir Computing?"*, you can adjust the hyperparameters below to see how the **Spectral Radius** and **Leak Rate** affect the model's ability to reconstruct the attractor.
""")

#  Sidebar Inputs 
st.sidebar.header("⚙️ ESN Hyperparameters")
st.sidebar.markdown("Adjust these to test the 'Edge of Stability'")

spectral_radius = st.sidebar.slider("Spectral Radius", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
leak_rate = st.sidebar.slider("Leak Rate", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
reservoir_size = st.sidebar.slider("Reservoir Size", min_value=100, max_value=2000, value=500, step=100)

st.sidebar.markdown("---")
st.sidebar.header("📈 Data Parameters")
num_steps = st.sidebar.number_input("Total Time Steps", min_value=1000, max_value=10000, value=5000)

# Main Execution  
if st.button("🚀 Train Model & Predict Chaos"):
    with st.spinner("Generating Lorenz data and training ESN..."):
        
        # 1. Generate Data
        data = generate_lorenz_data(num_steps=num_steps)
        X, Y = data[:-1], data[1:]
        
        train_size = int(len(X) * 0.8)
        X_train, Y_train = X[:train_size], Y[:train_size]
        X_test, Y_test = X[train_size:], Y[train_size:]
        
        # 2. Train Model
        model = EchoStateNetwork(
            input_size=3, 
            reservoir_size=reservoir_size, 
            spectral_radius=spectral_radius,
            leak_rate=leak_rate
        )
        model.fit(X_train, Y_train)
        
        # 3. Predict
        predictions = model.predict(initial_input=X_test[0], num_steps=len(X_test))
        mse = mean_squared_error(Y_test, predictions)
        
        # Display Metrics
        st.success(f"Model trained successfully! **Prediction Mean Squared Error:** {mse:.4f}")
        
        # 4. Interactive 3D Plot with Plotly
        st.subheader("Interactive 3D Phase Space: True vs. Predicted")
        
        fig = go.Figure()
        
        # True Trajectory
        fig.add_trace(go.Scatter3d(
            x=Y_test[:, 0], y=Y_test[:, 1], z=Y_test[:, 2],
            mode='lines',
            name='True Trajectory',
            line=dict(color='black', width=2)
        ))
        
        # Predicted Trajectory
        fig.add_trace(go.Scatter3d(
            x=predictions[:, 0], y=predictions[:, 1], z=predictions[:, 2],
            mode='lines',
            name='Predicted Trajectory',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)