# Chaotic System Forecasting with Echo State Networks

A Python ML pipeline for simulating and predicting the chaotic dynamics of the Lorenz Attractor using Reservoir Computing.

![ESN Dashboard Demo](https://github.com/user-attachments/assets/1e3f3c3e-29ad-470d-a913-fbfd7f6eed2a)

## Motivation
Predicting chaotic systems is a notoriously difficult task in Machine Learning due to their extreme sensitivity to initial conditions. This project demonstrates the implementation of an **Echo State Network (ESN)**—a type of Recurrent Neural Network (RNN)—built from scratch to forecast the multi-dimensional trajectory of the Lorenz system.

Unlike standard LSTMs or GRUs, the ESN utilizes a sparse, randomly connected hidden "reservoir" and only trains the linear readout weights, making it exceptionally fast to train while maintaining relatively high accuracy on chaotic time-series data.

Please see thesis for more mathematical detail
