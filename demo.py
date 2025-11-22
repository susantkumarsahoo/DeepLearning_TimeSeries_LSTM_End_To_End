import os
from src.pipelines.training_pipeline import run_data_ingestion_pipeline

dataset_path = r"C:\Users\TPWODL\New folder_Content\DeepLearning_TimeSeries_LSTM_End_To_End\data\raw\Energy Demand Hourly.csv"

artifact = run_data_ingestion_pipeline(dataset_path=dataset_path)
print(artifact)







