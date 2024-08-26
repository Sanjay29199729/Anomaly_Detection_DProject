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

# Setup GPU usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')

        # Set memory growth for the GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# List of CSV files
csv_files = ['/Users/sanjay/Desktop/Dproject/UNSW-NB15_1.csv',
             '/Users/sanjay/Desktop/Dproject/UNSW-NB15_2.csv',
             '/Users/sanjay/Desktop/Dproject/UNSW-NB15_3.csv',
             '/Users/sanjay/Desktop/Dproject/UNSW-NB15_4.csv']

# Specify the data types for problematic columns
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

# Load and concatenate CSV files into a single DataFrame with progress bar
dfs = []
for file in tqdm(csv_files, desc='Loading CSV files'):
    dfs.append(pd.read_csv(file, dtype=dtype_dict, header=None, low_memory=False))
    
print("Concatenating dataframes...")
df = pd.concat(dfs, ignore_index=True)

# Assign column names based on your schema or list of column names
df.columns = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 
              'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 
              'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 
              'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 
              'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 
              'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 
              'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label']

# Define categorical and numerical columns
cat_cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'service', 'attack_cat']
num_cols = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'Sload', 'Dload',
            'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
            'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt',
            'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
            'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
            'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']

# Handle missing values with mode for categorical and median for numerical
for col in tqdm(cat_cols, desc='Handling missing values and encoding categorical variables'):
    if col in df.columns:  # Check if column exists in DataFrame
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
            df[col] = df[col].astype(str)  # Ensure column is string type
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df[col].fillna(df[col].median(), inplace=True)
    else:
        print(f"Column '{col}' not found in DataFrame.")

# Handle missing values and convert numerical columns to float
for col in tqdm(num_cols, desc='Handling missing values and converting to float'):
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

# Normalize numerical features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Ensure label column exists
if 'label' not in df.columns:
    raise ValueError("The dataset must have a 'label' column defined as the target.")

# Separate features and target
print("Separating features and target...")
y = df.pop('label')  # Assuming 'label' is your target column

# Train-test split
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.6, random_state=42)

# Reshape for CNN and LSTM
X_train_cnn_lstm = np.expand_dims(X_train, axis=-1)
X_test_cnn_lstm = np.expand_dims(X_test, axis=-1)

# Define input shapes for different parts of the model
input_shape_autoencoder = (X_train.shape[1],)
input_shape_cnn = (X_train.shape[1], 1)  # Assuming 1D convolution

# Autoencoder for feature extraction with regularization and dropout
print("Building autoencoder...")
input_autoencoder = Input(shape=input_shape_autoencoder)
encoded = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(input_autoencoder)
encoded = Dropout(0.5)(encoded)
decoded = Dense(input_shape_autoencoder[0], activation='sigmoid')(encoded)
autoencoder = Model(input_autoencoder, decoded)

# CNN for spatial feature extraction with regularization and dropout
print("Building CNN...")
input_cnn = Input(shape=input_shape_cnn)
conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01))(input_cnn)
conv1 = Dropout(0.5)(conv1)
pool1 = MaxPooling1D(pool_size=2)(conv1)
flat1 = Flatten()(pool1)

# LSTM for temporal dependency modeling with regularization and dropout
print("Building LSTM...")
input_lstm = Input(shape=(X_train.shape[1], 1))  # Adjust input shape for LSTM
lstm1 = LSTM(50, kernel_regularizer=l2(0.01), return_sequences=True)(input_lstm)  # Start from input_lstm
lstm1 = Dropout(0.5)(lstm1)
lstm1 = LSTM(50, kernel_regularizer=l2(0.01))(lstm1)

# Merge the outputs of autoencoder, CNN, and LSTM
print("Merging model components...")
merged = concatenate([encoded, flat1, lstm1])
output = Dense(1, activation='sigmoid')(merged)

# Create the model
model = Model(inputs=[input_autoencoder, input_cnn, input_lstm], outputs=output)

# Compile the model
print("Compiling the model...")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping, model checkpoint, and TensorBoard
callbacks = [
    EarlyStopping(patience=3),
    ModelCheckpoint(filepath='best_model.keras', save_best_only=True),
    TensorBoard(log_dir=f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
]

# Train the model with added regularization and dropout
print("Training the model...")
history = model.fit(
    [X_train, X_train_cnn_lstm, X_train_cnn_lstm],
    y_train,
    validation_split=0.8,
    epochs=10,
    batch_size=256,
    callbacks=callbacks
)

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load the best model
print("Loading the best model...")
model.load_weights('best_model.keras')

# Evaluate the model
print("Evaluating the model...")
y_pred_prob = model.predict([X_test, X_test_cnn_lstm, X_test_cnn_lstm])
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate confusion matrix and classification report
print("Generating classification report...")
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Classification Report:")
print(cr)

# ROC curve and AUC
print("Generating ROC curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall curve
print("Generating Precision-Recall curve...")
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

plt.figure(figsize=(10, 7))
plt.plot(recall, precision)
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
