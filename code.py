import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

data = pd.read_csv('dataset.csv')
data

sample = pd.read_csv('sample_submission.csv')
sample

test = pd.read_csv('test.csv')
test

data.info()

data['subdivision']

data1 = data[['sale_date', 'sale_nbr', 'sale_warning',
       'join_status', 'join_year', 'latitude', 'longitude', 'area', 'city',
       'zoning', 'present_use', 'land_val', 'imp_val',
       'year_built', 'year_reno', 'sqft_lot', 'sqft', 'sqft_1', 'sqft_fbsmt',
       'grade', 'fbsmt_grade', 'condition', 'stories', 'beds', 'bath_full',
       'bath_3qtr', 'bath_half', 'garb_sqft', 'gara_sqft', 'wfnt', 'golf',
       'greenbelt', 'noise_traffic', 'view_rainier', 'view_olympics',
       'view_cascades', 'view_territorial', 'view_skyline', 'view_sound',
       'view_lakewash', 'view_lakesamm', 'view_otherwater', 'view_other',
       'submarket']]

y = data['sale_price']

test1 = test[['sale_date', 'sale_nbr', 'sale_warning',
       'join_status', 'join_year', 'latitude', 'longitude', 'area', 'city',
       'zoning', 'present_use', 'land_val', 'imp_val',
       'year_built', 'year_reno', 'sqft_lot', 'sqft', 'sqft_1', 'sqft_fbsmt',
       'grade', 'fbsmt_grade', 'condition', 'stories', 'beds', 'bath_full',
       'bath_3qtr', 'bath_half', 'garb_sqft', 'gara_sqft', 'wfnt', 'golf',
       'greenbelt', 'noise_traffic', 'view_rainier', 'view_olympics',
       'view_cascades', 'view_territorial', 'view_skyline', 'view_sound',
       'view_lakewash', 'view_lakesamm', 'view_otherwater', 'view_other',
       'submarket']]

from datetime import datetime
from transformers import AutoTokenizer

df = data1.copy()
cat_cols = ['join_status', 'city', 'zoning', 'submarket']

df['sale_date'] = pd.to_datetime(df['sale_date']).astype('int64') // 10**9
df['sale_warning'] = df['sale_warning'].str.strip().replace('', np.nan)
df['sale_warning'] = pd.to_numeric(df['sale_warning'], errors='coerce')

tr = test1.copy()

tr['sale_date'] = pd.to_datetime(tr['sale_date']).astype('int64') // 10**9
tr['sale_warning'] = tr['sale_warning'].str.strip().replace('', np.nan)
tr['sale_warning'] = pd.to_numeric(tr['sale_warning'], errors='coerce')

combined = pd.concat([df[cat_cols], tr[cat_cols]])

dummies = pd.get_dummies(combined, columns=cat_cols)

df1 = dummies.iloc[:len(df)].copy()
tr1 = dummies.iloc[len(df):].copy()

df = df.drop(columns=cat_cols)
tr = tr.drop(columns=cat_cols)

df = pd.concat([df, df1], axis=1)
tr = pd.concat([tr, tr1], axis=1)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def safe_tokenize(text):
    if pd.isna(text) or str(text).strip() == '':
        return []
    try:
        return tokenizer.encode(
            str(text),
            add_special_tokens=True,
            truncation=True,
            #max_length=128
        )
    except:
        return []

df['tokens'] = data['subdivision'].apply(safe_tokenize)
tr['tokens'] = test['subdivision'].apply(safe_tokenize)

df.shape, tr.shape

df.to_csv('df.csv', index=False)
tr.to_csv('tr.csv', index=False)
y.to_csv('y.csv', index=False)

df = pd.read_csv('df.csv')
tr = pd.read_csv('tr.csv')
y = pd.read_csv('y.csv').squeeze()

def handle_missing_values(df, y):
    if isinstance(df, pd.DataFrame):
        # For each column based on data type
        for col in df.columns:
            if df[col].dtype.kind in ['i', 'f']:  # Numeric columns
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            elif df[col].dtype.kind == 'b':  # Boolean values
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
            else:  # Strings and other types
                df[col] = df[col].fillna('missing')
    elif isinstance(df, np.ndarray):
        if df.dtype.kind in ['i', 'f']:
            df = np.nan_to_num(df, nan=np.nanmedian(df))

    # Handling target variable (y)
    if isinstance(y, (pd.Series, pd.DataFrame)):
        if y.dtype.kind in ['i', 'f']:
            y = y.fillna(y.median())
        elif y.dtype.kind == 'b':
            y = y.fillna(y.mode()[0])
        else:
            y = y.fillna('missing')
    elif isinstance(y, np.ndarray):
        if y.dtype.kind in ['i', 'f']:
            y = np.nan_to_num(y, nan=np.nanmedian(y))

    return df, y

df, y = handle_missing_values(df, y)

ts, y = handle_missing_values(tr, y)

coverage = 0.9
a = 1 - coverage

def loss(y_true, y_pred):
  """
  y_pred: [lower_bound, prediction, upper_bound]

  """
  lower = y_pred[:, 0]
  upper = y_pred[:, 2]

  width = upper - lower

  pen1 = tf.maximum(0.0, (2 / a) * (lower - y_true))
  pen2 = tf.maximum(0.0, (2 / a) * (y_true - upper))

  final_loss = width + pen1 + pen2

  return tf.reduce_mean(final_loss)

def interval_nw(input_shape):
  inputs = Input(shape=(input_shape,))

  x = Dense(128, activation='relu')(inputs)
  x = Dense(64, activation='elu')(x)
  x = Dense(8, activation='leaky_relu')(x)

  lower = Dense(1, activation='linear', name='lower')(x)
  point = Dense(1, activation='linear', name='point')(x)
  upper = Dense(1, activation='linear', name='upper')(x)

  outputs = Concatenate(axis=1)([lower, point, upper])

  model = Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer=Adam(0.0001), loss=loss)

  return model

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from ast import literal_eval

df['tokens'] = df['tokens'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
df['tokens'] = df['tokens'].apply(lambda x: x if isinstance(x, list) else [])

ts['tokens'] = ts['tokens'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
ts['tokens'] = ts['tokens'].apply(lambda x: x if isinstance(x, list) else [])

def prepare_data(df, ts):
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    bool_cols1 = ts.select_dtypes(include=['bool']).columns
    ts[bool_cols1] = ts[bool_cols1].astype(int)

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    X_numeric = df[numeric_cols].fillna(df[numeric_cols].median())
    numeric_cols1 = ts.select_dtypes(include=['number']).columns.tolist()
    ts_numeric = ts[numeric_cols1].fillna(ts[numeric_cols1].median())

    combined = np.vstack((X_numeric, ts_numeric))
    scaler = StandardScaler()
    scaler.fit(combined)
    X_numeric_scaled = scaler.transform(X_numeric)
    ts_numeric_scaled = scaler.transform(ts_numeric)

    return X_numeric_scaled, ts_numeric_scaled

X0, ts0 = prepare_data(df.copy(), ts.copy())

np.save('X0.npy', X0)
np.save('ts0.npy', ts0)

df.to_csv('df1.csv', index=False)
ts.to_csv('ts1.csv', index=False)
y.to_csv('y1.csv', index=False)

X0 = np.load('X0.npy')
ts0 = np.load('ts0.npy')

df = pd.read_csv('df1.csv')

ts = pd.read_csv('ts1.csv')

y = pd.read_csv('y1.csv').squeeze()

def prepare_data1(df, X0):
  if 'tokens' in df.columns:
        df['tokens'] = df['tokens'].apply(
            lambda x: literal_eval(x) if isinstance(x, str) else x
        )

        df['tokens'] = df['tokens'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        max_len = 128
        X_text = pad_sequences(
            df['tokens'].tolist(),
            #maxlen=max_len,
            padding='post',
            truncating='post',
            value=0
        )
        X_final = np.concatenate([X0, X_text], axis=1)
  else:
        X_final = X0

  return X_final

X_final = prepare_data1(df.copy(), X0.copy())

np.save('X_final.npy', X_final)

ts_final = prepare_data1(ts.copy(), ts0.copy())

np.save('ts_final.npy', ts_final)

y = np.array(y.copy(), dtype=np.float32)

np.save('y_final.npy', y)

X = np.load('X_final.npy')
ts = np.load('ts_final.npy')

y = np.load('y_final.npy')

print(f"Форма X: {X.shape}")
print(f"Форма y: {y.shape}")

model = interval_nw(X.shape[1])
history = model.fit(X, y, epochs=200, batch_size=32, validation_split=0.2, verbose=1)

predictions = model.predict(X)
lower_pred = predictions[:, 0]
point_pred = predictions[:, 1]
upper_pred = predictions[:, 2]

coverage = np.mean((y >= lower_pred) & (y <= upper_pred))
print(f"Фактическое покрытие интервала: {coverage:.2%}")

model.save_weights('model_weights.weights.h5')

X = np.load('X_final.npy')
ts = np.load('ts_final.npy')

loaded_model = interval_nw(X.shape[1])

loaded_model.load_weights('model_weights.weights.h5')

predictions1 = loaded_model.predict(ts)

lower_pred1 = predictions1[:, 0]
point_pred1 = predictions1[:, 1]
upper_pred1 = predictions1[:, 2]

lower_pred1

upper_pred1

test

final = test[['id']].copy()

final['pi_lower'] = lower_pred1
final['pi_upper'] = upper_pred1

final

final.to_csv('final.csv', index=False)
