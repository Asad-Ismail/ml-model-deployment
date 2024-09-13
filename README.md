# Machine Leanring Model deployment

Machine learning model deployment using AWS SageMaker. This repository explores different paradigms of deploying computer vision models, including building custom Docker containers from scratch and implementing AWS Batch Transform, real-time endpoints, and asynchronous endpoints.
Provides 

See blog post here for more details also
https://asad-ismail.github.io/posts/ml_deployment/


## Features

- Building custom Docker containers for machine learning models
- AWS Batch Transform: Provides a scalable and cost-effective solution for processing large volumes of data
- Real-time endpoints: Set up for immediate predictions
- Asynchronous endpoints: Handle large-scale inference requests efficiently
- Autoscaling configuration: For real-time and asynchronous endpoints to optimize resource usage

## Installation

To get started with this repository, follow these steps:

1. Clone the repository: 
'''
git clone https://github.com/your-username/ml-model-deployment.git
'''
2. Change into the project directory: 
```cd ml-model-deployment```
3. Install the required dependencies: ```pip install -r requirements.txt```

Note: Ensure you have the necessary AWS credentials and permissions configured for your environment if you are not running these notebooks from inside sagemaker.

## Usage

### AWS Sagemaker Real Time Endpoint

To deploy your model using Sagemaker real time endpoints, follow these steps:

1. ```cd sm_async_rt_endpoint```
2. Build docker file and push container to ECR run notebook 'build_Docker.ipynb'
3. Register model in a model group and get the model arm run notebook 'RegisterModel.ipynb'
4. Create RealTime Endpoint and test the Endpoint with sample data run notebook 'realtime-inference.ipynb'

### AWS Sagemaker Async Endpoint

To deploy your model as async endpoints, follow these steps:

1. ```cd sm_async_rt_endpoint```
2. Build docker file and push container to ECR run notebook 'build_Docker.ipynb'
3. Register model in a model group and get the model arm run notebook 'RegisterModel.ipynb'
4. Create Async Endpoint and test the Endpoint with sample data run notebook 'async-inference.ipynb'

### AWS Sagemaker Batch Transform

To process the data in batch transform using your model as async endpoints, follow these steps:

1. ```cd sm_batch_transform```
2. Build docker file and push container to ECR run notebook 'build_Docker.ipynb'
3. Register model in a model group and get the model arm run notebook 'RegisterModel.ipynb'
4. Run batch transform job using your model run notebook 'batch_transform.ipynb'


### Additional Commnets/ Suggestions for putting the model in productos
- **Use Model Groups**: Organize your models using SageMaker Model Groups, and include metadata like accuracy on the test set. This practice is demonstrated in the notebooks.

- **Model Approval Process**: Typically, models are approved by other team members or stakeholders after reviewing the model's performance and suitability for production.

- **Autoscaling**: Implement autoscaling for asynchronous endpoints. For instance, when there is a burst of requests, the service can scale up, and when there are no requests for some time, it can scale down, possibly to zero instances if your application can tolerate the cold start latency.

- **Notification Services**: Use notification services like Amazon SNS to receive alerts when asynchronous jobs are completed, instead of polling in a loop as shown in the example notebooks.

- **Traffic Shifting Strategies**: In production environments, it's common practice not to divert all traffic to a new model at once. Instead, techniques like A/B testing, canary deployments, or blue/green deployments are used to gradually redirect traffic to the new model. This allows you to monitor the new model's performance in a controlled manner and quickly roll back if any issues are detected.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.


