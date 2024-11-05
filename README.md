# Project Overview

This project aims to address the challenge of monitoring vehicle health and predicting faults in automotive internal combustion (IC) engines. Using machine learning techniques, we explore fault classification and prediction, utilizing the EngineFaultDB dataset. The project’s ultimate goal is to provide a model capable of accurately predicting faults to support preventative maintenance.

# Problem Statement

Traditional methods for detecting faults in automotive engines lack efficiency and accuracy. This project introduces AI and ML techniques to classify and predict engine faults, moving beyond conventional methods to improve reliability.

# Dataset

The project utilizes the EngineFaultDB dataset, which contains 55,999 entries and 14 distinct variables. Data has been categorized into four main fault types. Challenges identified with synthetic datasets (low realism and high potential for overfitting) are also addressed in this project.

## Dataset Source

	M. Vergara, L. Ramos, and N. D. Rivera-Campoverde, “EngineFaultDB: A Novel Dataset for Automotive Engine Fault Classification and Baseline Results,” IEEE Access, vol. 11, pp. 126155-126171, 2023.

# Methodology

Data Preprocessing

	•	The dataset was pre-processed to handle missing values and to prepare it for training machine learning models.
	•	Visualizations like heatmaps and boxplots were used to understand correlations and outliers in the data.

# Algorithms Used

We experimented with multiple machine learning algorithms:

	1.	Random Forest
	2.	Support Vector Machine (SVM)
	3.	K-Nearest Neighbors (KNN)
	4.	Feedforward Neural Network

# Improvements on Previous Work

	•	Models were tuned to improve accuracy and reduce overfitting.
	•	Advanced models, CNN (Convolutional Neural Network) and DNN (Deep Neural Network), were introduced, yielding better results.

# Results

	•	CNN Model Accuracy: 76.5%
	•	DNN Model Accuracy: 75.5%

These improvements demonstrate the potential of using AI to effectively monitor and predict engine faults.
