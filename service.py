
import json
import numpy as np
import bentoml
import pandas as pd
import pydantic
from bentoml.io import JSON, PandasSeries, PandasDataFrame


runner = bentoml.sklearn.get('diabetes_pred_elastic:latest').to_runner()
svc = bentoml.Service("diabetes_pred_elastic", runners=[runner])


@svc.api(
    input=PandasDataFrame(),
    output=JSON(),
    route='v1/predict/'
)
def predict(dia_df: pd.DataFrame) -> json:
    dia_df.columns = ['age', 'sex', 'bmi',
                        'bp', 's1', 's2',	's3', 's4', 's5', 's6']
    pred = runner.run(dia_df.astype(float))
    return {'pred': pred}
