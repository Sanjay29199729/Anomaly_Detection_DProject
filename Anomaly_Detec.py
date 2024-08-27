import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, LSTM, Flatten, concatenate, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Setup GPU usage if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')

        # Set memory growth to avoid memory allocation issues
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# List of CSV files to be loaded
csv_files = [
    '/Users/sanjay/Desktop/Dproject/UNSW-NB15_1.csv',
    '/Users/sanjay/Desktop/Dproject/UNSW-NB15_2.csv',
    '/Users/sanjay/Desktop/Dproject/UNSW-NB15_3.csv',
    '/Users/sanjay/Desktop/Dproject/UNSW-NB15_4.csv'
]

# Specify data types for specific columns to ensure proper loading
dtype_dict = {
    'srcip': str,
    'sport': str,
    'dstip': str,
    'dsport': str,
    'proto': str,
    'state': str,
    'service': str,
    'attack_cat': str
}

# Load and concatenate CSV files into a single DataFrame with a progress bar
dfs = []
for file in tqdm(csv_files, desc='Loading CSV files'):
    dfs.append(pd.read_csv(file, dtype=dtype_dict, header=None, low_memory=False))

# Concatenate all loaded dataframes into one
print("Concatenating dataframes...")
df = pd.concat(dfs, ignore_index=True)

# Assigning column names based on the schema of the dataset
df.columns = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 
    'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 
    'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 
    'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 
    'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 
    'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
]

# Define categorical and numerical columns separately
cat_cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'service', 'attack_cat']
num_cols = [
    'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'Sload', 'Dload',
    'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
    'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt',
    'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
    'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
    'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm'
]

# Handle missing values and encode categorical variables
for col in tqdm(cat_cols, desc='Handling missing values and encoding categorical variables'):
    if col in df.columns:  # Ensure the column exists
        if df[col].dtype == 'object':  # If column is categorical
            df[col].fillna(df[col].mode()[0], inplace=True)  # Fill missing values with mode
            df[col] = df[col].astype(str)  # Ensure the column is of string type
            df[col] = LabelEncoder().fit_transform(df[col])  # Encode categorical values
        else:
            df[col].fillna(df[col].median(), inplace=True)  # Fill missing values with median for numerical
    else:
        print(f"Column '{col}' not found in DataFrame.")

# Handle missing values and convert numerical columns to float
for col in tqdm(num_cols, desc='Handling missing values and converting to float'):
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coerce errors to NaN
    df[col].fillna(df[col].median(), inplace=True)  # Fill missing values with the median

# Normalize numerical features for better model performance
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Ensure the target label column exists in the dataset
if 'label' not in df.columns:
    raise ValueError("The dataset must have a 'label' column defined as the target.")

# Separate features (X) and target (y)
print("Separating features and target...")
y = df.pop('label')  # Separate the target variable

# Split data into training and testing sets (60% for testing)
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.6, random_state=42)

# Reshape data for CNN and LSTM input
X_train_cnn_lstm = np.expand_dims(X_train, axis=-1)  # Add a new axis for CNN and LSTM compatibility
X_test_cnn_lstm = np.expand_dims(X_test, axis=-1)

# Define input shapes for different parts of the model
input_shape_autoencoder = (X_train.shape[1],)  # Shape for autoencoder input
input_shape_cnn = (X_train.shape[1], 1)  # Shape for CNN input (assuming 1D convolution)

# Building an autoencoder for feature extraction with regularization and dropout
print("Building autoencoder...")
input_autoencoder = Input(shape=input_shape_autoencoder)  # Input layer
encoded = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(input_autoencoder)  # Encoding with L2 regularization
encoded = Dropout(0.5)(encoded)  # Dropout for regularization
decoded = Dense(input_shape_autoencoder[0], activation='sigmoid')(encoded)  # Decoding layer
autoencoder = Model(input_autoencoder, decoded)  # Define autoencoder model

# Building a CNN for spatial feature extraction with regularization and dropout
print("Building CNN...")
input_cnn = Input(shape=input_shape_cnn)  # Input layer for CNN
conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01))(input_cnn)  # Convolution layer
conv1 = Dropout(0.5)(conv1)  # Dropout for regularization
pool1 = MaxPooling1D(pool_size=2)(conv1)  # Max pooling to reduce spatial dimensions
flat1 = Flatten()(pool1)  # Flatten the output for dense layers

# Building an LSTM for temporal dependency modeling with regularization and dropout
print("Building LSTM...")
input_lstm = Input(shape=(X_train.shape[1], 1))  # Input layer for LSTM
lstm1 = LSTM(50, kernel_regularizer=l2(0.01), return_sequences=True)(input_lstm)  # First LSTM layer with L2 regularization
lstm1 = Dropout(0.5)(lstm1)  # Dropout for regularization
lstm1 = LSTM(50, kernel_regularizer=l2(0.01))(lstm1)  # Second LSTM layer

# Merging the outputs of the autoencoder, CNN, and LSTM
print("Merging model components...")
merged = concatenate([encoded, flat1, lstm1])  # Combine features from all models
output = Dense(1, activation='sigmoid')(merged)  # Output layer with sigmoid activation for binary classification

# Create the final model combining autoencoder, CNN, and LSTM
model = Model(inputs=[input_autoencoder, input_cnn, input_lstm], outputs=output)

# Compile the model with Adam optimizer and binary cross-entropy loss
print("Compiling the model...")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks for early stopping, saving the best model, and TensorBoard logging
callbacks = [
    EarlyStopping(patience=3),  # Stop training if no improvement after 3 epochs
    ModelCheckpoint(filepath='best_model.keras', save_best_only=True),  # Save the best model
    TensorBoard(log_dir=f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")  # Log training process for TensorBoard
]

# Train the model with training data, using validation split
print("Training the model...")
history = model.fit(
    [X_train, X_train_cnn_lstm, X_train_cnn_lstm],  # Inputs for autoencoder, CNN, and LSTM
    y_train,  # Target values
    validation_split=0.8,  # 80% of the training data will be used for validation
    epochs=10,  # Number of epochs
    batch_size=256,  # Batch size
    callbacks=callbacks  # Callbacks defined earlier
)

# Plot training and validation accuracy over epochs
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load the best model saved during training
print("Loading the best model...")
model.load_weights('best_model.keras')

# Evaluate the model on test data
print("Evaluating the model...")
y_pred_prob = model.predict([X_test, X_test_cnn_lstm, X_test_cnn_lstm])  # Predict probabilities on test data
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Generate confusion matrix and classification report
print("Generating classification report...")
cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix
cr = classification_report(y_test, y_pred)  # Generate classification report

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Classification Report:")
print(cr)

# Plot ROC curve and calculate AUC
print("Generating ROC curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)  # Compute ROC curve
roc_auc = auc(fpr, tpr)  # Calculate AUC

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Plot Precision-Recall curve
print("Generating Precision-Recall curve...")
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)  # Compute Precision-Recall curve

plt.figure(figsize=(10, 7))
plt.plot(recall, precision)
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
