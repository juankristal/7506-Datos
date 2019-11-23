from category_encoders import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_train_simple_pre_pipeline():
    columns_pipe = ColumnTransformer(transformers = [
        ('nan_to_mean', SimpleImputer(strategy = 'mean'), ['metrostotales', 'metroscubiertos', 'antiguedad']),
        ('nan_to_cero', SimpleImputer(strategy = 'constant', fill_value = 0), ['habitaciones', 'banos', 'garages'])
    ])

    pre_processor_pipe = Pipeline(steps =[
        ('ordinal_encoder', OrdinalEncoder(cols = ['tipodepropiedad', 'provincia', 'ciudad'])),
        ('columns_pipe', columns_pipe)
    ])
    return pre_processor_pipe