from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, ZeroPadding1D
from config import SEED, CHARACTER_DICT, MAX_SEQUENCE_LENGTH

# ==========================================
# 2. CNN Architecture Definition
# ==========================================
def build_witten_model():
    character_to_index_len = len(CHARACTER_DICT)
    model = Sequential()
    
    # Padding layer
    model.add(ZeroPadding1D(5, input_shape=(MAX_SEQUENCE_LENGTH, character_to_index_len + 1)))
    
    # First convolution + pooling
    model.add(Conv1D(64, kernel_size=5, strides=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    
    # Second convolution + pooling
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Flatten + Dense
    model.add(Flatten())
    model.add(Dropout(0.5, seed=SEED))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='linear')) # Output continuous MIC values
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model