import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Sample dataset
df = pd.DataFrame({
    'text': [
        "Trump’s tariffs are disastrous!",
        "Great move by Trump to protect US jobs.",
        "Markets crash after new Trump policy.",
        "Tariffs finally show strength. Good work!"
    ],
    'retweets': [1200, 40, 1500, 30],
    'likes': [3000, 100, 2500, 60],
    'followers': [100000, 2000, 80000, 1800],
})

# Label: retweets > 500 = viral
df['label'] = (df['retweets'] > 500).astype(int)

# Metadata features
metadata = df[['followers', 'likes']]
scaler = StandardScaler()
metadata_scaled = scaler.fit_transform(metadata)

# Text vectorization
vectorizer = TextVectorization(max_tokens=1000, output_mode='tf-idf')
vectorizer.adapt(df['text'].values)
text_vectors = vectorizer(df['text'].values)

# Train/test split
X_text_train, X_text_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
    text_vectors.numpy(), metadata_scaled, df['label'], test_size=0.2)

# Build model
text_input = Input(shape=(X_text_train.shape[1],), name="text_input")
meta_input = Input(shape=(X_meta_train.shape[1],), name="meta_input")

# Combine both
x = Concatenate()([text_input, meta_input])
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[text_input, meta_input], outputs=output)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit([X_text_train, X_meta_train], y_train, epochs=10, batch_size=2, validation_split=0.2)

# Predict
predictions = model.predict([X_text_test, X_meta_test])
print("Predicted viral probabilities:", predictions.flatten())