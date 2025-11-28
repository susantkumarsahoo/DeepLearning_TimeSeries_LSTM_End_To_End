# DeepLearning_TimeSeries_LSTM_End_To_End
LSTM uses a Long Short-Term Memory (LSTM) network to process historical data and predict future values directly, from raw input to final forecast, without manual feature engineering between layers.


## git add .
## git commit -m "file uplode"
## git push origin main
## git pull
## git status


## dvc add
## git commit
## dvc push
## dvc pull

## dvc init
## git status
## git commit -m "Initialize DVC"
## dvc add artifacts/
## dvc status


dvc init
git add .
git commit -m "Initialize DVC"
git add .
git commit -m "Run DVC pipeline"
dvc dag
dvc repro




  feature_transformer:
    cmd: python src/pipelines/feature_transformer_pipeline.py
    deps:
      - src/constants/paths.py
      - src/utils/feature_transformer_helper.py
      - src/entity/components_config_entity.py
      - src/components/feature_transformer.py
      - artifacts/feature_engineering/test_features.csv
      - artifacts/feature_engineering/train_features.csv
    outs:
      - artifacts/feature_transformer/test_features.csv
      - artifacts/feature_transformer/train_features.csv