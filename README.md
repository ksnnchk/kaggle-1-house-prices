# Real Estate Price Prediction with Prediction Intervals

This project implements a neural network model for real estate price prediction with prediction intervals, providing both point estimates and uncertainty quantification.

## Overview

The model predicts house sale prices using various property features and provides prediction intervals to quantify uncertainty in the predictions. This is particularly valuable for real estate applications where understanding the range of possible prices is as important as the point estimate.

## Dataset

The dataset contains real estate transaction records with the following features:

### Key Features:
- **sale_date**: Date of sale (converted to timestamp)
- **sale_nbr**: Sale number identifier
- **sale_warning**: Sale warning indicator
- **join_status**, **join_year**: Property joining information
- **latitude**, **longitude**: Geographic coordinates
- **area**, **city**: Location information
- **zoning**: Zoning classification
- **present_use**: Current property use
- **land_val**, **imp_val**: Land and improvement values
- **year_built**, **year_reno**: Construction and renovation years
- **sqft_lot**, **sqft**, **sqft_1**, **sqft_fbsmt**: Various square footage measurements
- **grade**, **fbsmt_grade**: Quality grades
- **condition**: Property condition
- **stories**: Number of stories
- **beds**, **bath_full**, **bath_3qtr**, **bath_half**: Bedroom and bathroom counts
- **garb_sqft**, **gara_sqft**: Garage and garbage area measurements
- **Various view indicators**: (wfnt, golf, greenbelt, noise_traffic, view_*)
- **submarket**: Real estate submarket classification
- **subdivision**: Text description of subdivision (tokenized using BERT)

### Target Variable:
- **sale_price**: The sale price to be predicted

## Model Architecture

The model uses a custom neural network with the following structure:

```python
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
```

## Custom Loss Function

The model uses a custom loss function designed to optimize prediction intervals:

```python
def loss(y_true, y_pred):
    """
    Custom loss function for prediction intervals
    y_pred: [lower_bound, prediction, upper_bound]
    """
    lower = y_pred[:, 0]
    upper = y_pred[:, 2]
    width = upper - lower
    
    pen1 = tf.maximum(0.0, (2 / a) * (lower - y_true))
    pen2 = tf.maximum(0.0, (2 / a) * (y_true - upper))
    
    final_loss = width + pen1 + pen2
    return tf.reduce_mean(final_loss)
```

## Data Preprocessing

1. **Temporal Data**: Sale dates converted to timestamps
2. **Categorical Variables**: One-hot encoded using `pd.get_dummies()`
3. **Text Data**: Subdivision descriptions tokenized using BERT tokenizer
4. **Missing Values**: Handled with median imputation for numerical data and mode for categorical data
5. **Feature Scaling**: StandardScaler applied to numerical features
6. **Sequence Padding**: Token sequences padded to consistent length

## Usage

### Training:
```python
model = interval_nw(X.shape[1])
history = model.fit(X, y, epochs=200, batch_size=32, validation_split=0.2, verbose=1)
```

### Prediction:
```python
predictions = model.predict(ts)
lower_pred = predictions[:, 0]  # Lower bound of prediction interval
point_pred = predictions[:, 1]  # Point estimate
upper_pred = predictions[:, 2]  # Upper bound of prediction interval
```

### Evaluation:
```python
coverage = np.mean((y >= lower_pred) & (y <= upper_pred))
print(f"Prediction interval coverage: {coverage:.2%}")
```

## Output

The model generates predictions with prediction intervals in the format:
- `id`: Property identifier
- `pi_lower`: Lower bound of prediction interval
- `pi_upper`: Upper bound of prediction interval

## Dependencies

- pandas
- numpy
- tensorflow
- transformers (Hugging Face)
- scikit-learn
- datetime

## Files

- `dataset.csv`: Training data
- `test.csv`: Test data
- `sample_submission.csv`: Sample submission format
- `model_weights.weights.h5`: Saved model weights
- `final.csv`: Final predictions with intervals
