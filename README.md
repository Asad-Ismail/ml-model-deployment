# ml-model-deployment
Machine learning model deployment using AWS. Different paradigm of deploying computer vision models

Provides 
## Repository Name

ML Model Deployment

## Description

This repository provides a comprehensive guide on deploying machine learning models using AWS. It focuses on a different paradigm of deploying computer vision models and demonstrates how to build a custom container from scratch. It also covers the implementation of AWS Batch Transform, real-time endpoints, and asynchronous endpoints.

## Features

- Building custom containers for machine learning models
- Demonstrating the use of AWS Batch Transform. It provides a scalable and cost-effective solution for processing large volumes 
- Setting up real-time endpoints for real-time predictions
- Creating asynchronous endpoints for handling large-scale inference requests
- Setting autoscaling for realtime and async endpoints

## Installation

To get started with this repository, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/ml-model-deployment.git`
2. Change into the project directory: `cd ml-model-deployment`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage


### AWS Sagemaker Real Time Endpoint

To deploy your model using Sagemaker real time endpoints, follow these steps:

1. cd sm_async_rt_endpoint
2. Build docker file and push container to ECR run notebook 'build_Docker.ipynb'
3. Create RealTime Endpoint and test the Endpoint with sample data run notebook 'realtime-inference.ipynb'

### AWS Sagemaker Async Endpoint

To set up async endpoints for your model, follow these steps:

1. Open the `realtime_endpoints.ipynb` notebook
2. Follow the instructions provided in the notebook to deploy and test the real-time endpoint

### Asynchronous Endpoints

To create asynchronous endpoints for handling large-scale inference requests, follow these steps:

1. Open the `async_endpoints.ipynb` notebook
2. Follow the instructions provided in the notebook to deploy and test the asynchronous endpoint

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.


