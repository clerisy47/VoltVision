# Energy Demand Forecasting System

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

[Presentation link for the project](https://www.canva.com/design/DAF6OD-oHkk/MdA1Nltv_iqwEGbWbSLrtA/edit?utm_content=DAF6OD-oHkk&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

[Project Documentation](./Nansoscopic_Innovators_Documentation.pdf)

## Problem Statement

The energy industry is navigating a transformative phase characterized by rapid modernization and technological advancements. Infrastructure upgrades, integration of intermittent renewable energy sources, and evolving consumer demands are reshaping the sector. However, the progress comes with challenges like volatile supply, demand, and prices, rendering the future less predictable. Traditional business models are also being fundamentally challenged. In this competitive and dynamic landscape, accurate decision-making is pivotal.

Stakeholders in the energy industry heavily rely on probabilistic forecasts to navigate this uncertain future. Therefore, innovative and precise forecasting methods are essential to assist stakeholders in making strategic decisions amidst the shifting energy landscape.

## Python Requirements

### Installation

To set up the project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/zenml-io/zenml-projects.git
    cd zenml-projects/customer-satisfaction
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Launch the ZenML Server and Dashboard:

    ```bash
    pip install zenml["server"]
    zenml up
    ```

4. Install MLflow integration for deployment:

    ```bash
    zenml integration install mlflow -y
    ```

5. Configure ZenML stack with MLflow components:

    ```bash
    zenml experiment-tracker register mlflow_tracker --flavor=mlflow
    zenml model-deployer register mlflow --flavor=mlflow
    zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
    ```

### Training Pipeline

The training pipeline includes the following steps:

- `ingest_data`: Ingests and creates a `DataFrame` from the data.
- `clean_data`: Cleans the data and removes unwanted columns.
- `train_model`: Trains the model and saves it using MLflow autologging.
- `evaluation`: Evaluates the model and saves metrics using MLflow autologging.

### Deployment Pipeline

The deployment pipeline extends the training pipeline and implements a continuous deployment workflow. Additional steps include:

- `deployment_trigger`: Checks whether the newly trained model meets deployment criteria.
- `model_deployer`: Deploys the model as a service using MLflow (if criteria is met).

The deployment pipeline uses ZenML's MLflow tracking integration for logging hyperparameter values, the trained model, and model evaluation metrics into the local MLflow backend. It launches a local MLflow deployment server to serve the latest model if its accuracy is above a configured threshold.

A Streamlit application is deployed to consume the latest model service asynchronously from the pipeline logic.

![training_and_deployment_pipeline](assets/pipeline.png)




## Diving into the Code

Run the training pipeline:

```bash
python run_pipeline.py
```

Run the continuous deployment pipeline:

```bash
python run_deployment.py
```

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

## FAQ

1. **Error: No Step found for the name mlflow_deployer**:

   Solution: Delete the artifact store and rerun the pipeline. Obtain the artifact store location:

   ```bash
   zenml artifact-store describe
   ```

   Delete the artifact store:

   ```bash
   rm -rf PATH
   ```

2. **Error: No Environment component with name mlflow is currently registered**:

   Solution: Install the MLflow integration:

   ```bash
   zenml integration install mlflow -y
   ```
