import os
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Ensure determinism
SEED = 123456
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Set the random seed for all required libraries
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# Generate some dummy data
X = np.random.rand(1000, 10)  # 1000 samples, 10 features
y = 2 * X[:, 0] + 1 + np.random.normal(0, 0.1, 1000)

# Split the data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(train_X[0])
# Define the model architecture (minimalistic version of your project)
def build_model():
    inputs = Input(shape=(train_X.shape[1],), name='input_layer')
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu')(inputs)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5, seed=SEED)(x, training=False)  # Ensure dropout is not applied during inference
    outputs = Dense(1, activation=None)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build and train the model
model = build_model()

# Train the model
history = model.fit(train_X, train_y, batch_size=32, epochs=2, verbose=2)

# Evaluate the model
test_loss = model.evaluate(test_X, test_y, verbose=0)
print(f"Test Loss: {test_loss}")
