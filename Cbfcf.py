import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

# Collaborative Filtering (CF) Model
class CFModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size, num_numerical_features):
        super(CFModel, self).__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_size)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(embedding_size, activation='relu')
        self.num_features_dense = tf.keras.layers.Dense(embedding_size, activation='relu')
        self.num_numerical_features = num_numerical_features

    def call(self, inputs):
        user_ids, item_ids, num_features = inputs[:, 0], inputs[:, 1], inputs[:, 2:]
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        user_embedding = self.flatten(user_embedding)
        item_embedding = self.flatten(item_embedding)
        num_features_output = self.num_features_dense(num_features[:, :self.num_numerical_features])
        concatenated = tf.concat([user_embedding, item_embedding, num_features_output], axis=-1)
        output = self.dense(concatenated)
        return output

# Content-Based Filtering (CBF) Model
class CBFModel(tf.keras.Model):
    def __init__(self, bert_hidden_size, num_features, num_numerical_features):
        super(CBFModel, self).__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.dense = tf.keras.layers.Dense(bert_hidden_size, activation='relu')
        self.num_features_dense = tf.keras.layers.Dense(bert_hidden_size, activation='relu')
        self.num_features = num_features
        self.num_numerical_features = num_numerical_features

    def call(self, inputs):
        input_ids, attention_mask, num_features = inputs[:, :self.num_features], inputs[:, self.num_features:-self.num_numerical_features], inputs[:, -self.num_numerical_features:]
        outputs = self.bert(input_ids, attention_mask=attention_mask)[0]
        pooled_output = self.pooling(outputs)
        num_features_output = self.num_features_dense(num_features)
        combined_output = tf.concat([pooled_output, num_features_output], axis=1)
        return self.dense(combined_output)

# Prediction Layer
class PredictionLayer(tf.keras.layers.Layer):
    def __init__(self, num_factors, bert_hidden_size):
        super(PredictionLayer, self).__init__()
        self.num_factors = num_factors
        self.bert_hidden_size = bert_hidden_size
        self.linear_cf = tf.keras.layers.Dense(1)
        self.linear_cbf = tf.keras.layers.Dense(1)
        self.linear_num_features = tf.keras.layers.Dense(1)

    def call(self, inputs):
        cf_output, cbf_output, num_features = inputs
        cf_output = tf.squeeze(cf_output, axis=1)  # Remove the last dimension of cf_output
        cbf_output = tf.reduce_mean(cbf_output, axis=1)  # Average pooling along the sequence length of cbf_output
        num_features_output = self.linear_num_features(num_features)
        cf_output = self.linear_cf(cf_output)
        cbf_output = self.linear_cbf(cbf_output)
        output = cf_output + cbf_output + num_features_output
        return output

# Random Data Generation
num_users = 1000
num_items = 5000
num_features = 10
num_numerical_features = 5

cf_users = np.random.randint(0, num_users, size=(4000,))
cf_items = np.random.randint(0, num_items, size=(4000,))
cf_ratings = np.random.randint(0, 5, size=(4000, 1))
cf_num_features = np.random.rand(num_items, num_features + num_numerical_features)

cbf_texts = np.random.randint(0, 100, size=(500,))
cbf_num_features = np.random.rand(num_items, num_features + num_numerical_features)

# Collaborative Filtering (CF) Training
cf_model = CFModel(num_users=num_users, num_items=num_items, embedding_size=32, num_numerical_features=num_numerical_features)
optimizer_cf = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def cf_train_step(cf_users, cf_items, cf_num_features, cf_ratings):
    with tf.GradientTape() as tape:
        cf_inputs = tf.concat([cf_users, cf_items, cf_num_features], axis=-1)
        cf_outputs = cf_model(cf_inputs)
        cf_loss = tf.keras.losses.MeanSquaredError()(cf_ratings, cf_outputs)
    cf_gradients = tape.gradient(cf_loss, cf_model.trainable_variables)
    optimizer_cf.apply_gradients(zip(cf_gradients, cf_model.trainable_variables))

# Content-Based Filtering (CBF) Training
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

encoded_inputs = tokenizer(cbf_texts.tolist(), padding=True, truncation=True, return_tensors='tf')
input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']

cbf_model = CBFModel(bert_hidden_size=64, num_features=num_features, num_numerical_features=num_numerical_features)
cbf_outputs = cbf_model([input_ids, attention_mask, cbf_num_features])

# Prepare the data for the prediction layer
cf_outputs = cf_model(tf.concat([cf_users, cf_items, cf_num_features], axis=-1))
cf_outputs = tf.expand_dims(cf_outputs, axis=1)
cbf_outputs = tf.expand_dims(cbf_outputs, axis=1)
cbf_num_features = tf.convert_to_tensor(cbf_num_features, dtype=tf.float32)

# Initialize the HybridBERT4Rec model
hybrid_model = PredictionLayer(num_factors=10, bert_hidden_size=64)
optimizer_hybrid = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def hybrid_train_step(cf_outputs, cbf_outputs, cbf_num_features, cf_ratings):
    with tf.GradientTape() as tape:
        hybrid_predictions = hybrid_model([cf_outputs, cbf_outputs, cbf_num_features])
        hybrid_loss = tf.keras.losses.MeanSquaredError()(cf_ratings, hybrid_predictions)
    hybrid_gradients = tape.gradient(hybrid_loss, hybrid_model.trainable_variables)
    optimizer_hybrid.apply_gradients(zip(hybrid_gradients, hybrid_model.trainable_variables))

# Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    cf_train_step(cf_users, cf_items, cf_num_features, cf_ratings)
    hybrid_train_step(cf_outputs, cbf_outputs, cbf_num_features, cf_ratings)
    print(f"Epoch {epoch+1}/{num_epochs} - CF Loss: {cf_loss:.4f} - Hybrid Loss: {hybrid_loss:.4f}")



