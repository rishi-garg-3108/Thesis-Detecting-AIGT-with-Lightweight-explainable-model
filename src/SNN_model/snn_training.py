import random
import sys
import logging
import argparse
import os
import time
import json
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import shuffle
from transformers import GPT2Tokenizer
from tensorflow.keras.layers import Layer, Input, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# Setup logging to print to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def set_seed(seed: int):
    """Sets random seeds for reproducibility across random, numpy, and tensorflow.

    Args:
        seed (int): The seed value to set for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logging.info(f"Seed set to: {seed}")

# Hyperparameters
num_layers = 4
d_model = 128  # Embedding dimension
num_heads = 4  # Number of attention heads
dff = 512  # Feed forward network dimension
input_vocab_size = 50257
min_token_count = 30
max_seq_len = 150  # Maximum sequence length of tweet after padding
dropout_rate = 0.1
learning_rate = 0.0006
output_size = 128
margin = 2

class TweetDataset:
    def __init__(self, file_path, seed, tokenizer_name="gpt2", min_token_count=30, max_seq_len=150):
        """Initializes the TweetDataset class for loading and preprocessing tweet data.

        Args:
            file_path (str): Path to the dataset (JSON format).
            seed (int): Seed for random operations.
            tokenizer_name (str): The name of the tokenizer to use (default: "gpt2").
            min_token_count (int): Minimum number of tokens required for a tweet.
            max_seq_len (int): Maximum sequence length for padding.
        """
        logging.info(f"Initializing TweetDataset with file: {file_path}, seed: {seed}")
        self.file_path = file_path
        self.seed = seed
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.min_token_count = min_token_count
        self.max_seq_len = max_seq_len
        self.trainList_orig_encoded_padded, self.testList_orig_encoded_padded, \
        self.trainList_fake_encoded_padded, self.testList_fake_encoded_padded = self._preprocess_data()

    def _preprocess_data(self):
        """Preprocesses the tweet data by splitting, tokenizing, filtering, and padding.

        Returns:
            Tuple: Training and test sets for original and fake tweets (padded sequences).
        """
        logging.info("Preprocessing data...")
        tweets = pd.read_json(self.file_path, lines=True)
        tweets_map = {k: g["tweets"].tolist() for k, g in tweets.groupby("artificial")}
        logging.info(f"Number of original tweets: {len(tweets_map[0])}")
        logging.info(f"Number of fake tweets: {len(tweets_map[1])}")

        self.original_tweets = tweets_map[0]
        self.fake_tweets = tweets_map[1]

        # Step 1: Split data into train and test sets
        self.trainList_orig, self.testList_orig = self._split_data(self.original_tweets)
        self.trainList_fake, self.testList_fake = self._split_data(self.fake_tweets)
        logging.info(f"Training original tweets: {len(self.trainList_orig)}")
        logging.info(f"Testing original tweets: {len(self.testList_orig)}")
        logging.info(f"Training fake tweets: {len(self.trainList_fake)}")
        logging.info(f"Testing fake tweets: {len(self.testList_fake)}")

        # Step 2: Encode and filter tweets
        trainList_orig_encoded, testList_orig_encoded = self._encode_and_filter_tweets(self.trainList_orig, self.testList_orig)
        trainList_fake_encoded, testList_fake_encoded = self._encode_and_filter_tweets(self.trainList_fake, self.testList_fake)
        logging.info(f"Filtered training original tweets: {len(trainList_orig_encoded)}")
        logging.info(f"Filtered testing original tweets: {len(testList_orig_encoded)}")
        logging.info(f"Filtered training fake tweets: {len(trainList_fake_encoded)}")
        logging.info(f"Filtered testing fake tweets: {len(testList_fake_encoded)}")

        # Log discarded tweets
        logging.info(f"Discarded training original tweets: {len(self.trainList_orig) - len(trainList_orig_encoded)}")
        logging.info(f"Discarded testing original tweets: {len(self.testList_orig) - len(testList_orig_encoded)}")
        logging.info(f"Discarded training fake tweets: {len(self.trainList_fake) - len(trainList_fake_encoded)}")
        logging.info(f"Discarded testing fake tweets: {len(self.testList_fake) - len(testList_fake_encoded)}")

        # Pad sequences to the same length
        trainList_orig_encoded_padded = self._pad_sequences(trainList_orig_encoded)
        testList_orig_encoded_padded = self._pad_sequences(testList_orig_encoded)
        trainList_fake_encoded_padded = self._pad_sequences(trainList_fake_encoded)
        testList_fake_encoded_padded = self._pad_sequences(testList_fake_encoded)

        return trainList_orig_encoded_padded, testList_orig_encoded_padded, trainList_fake_encoded_padded, testList_fake_encoded_padded

    def _split_data(self, tweets, train_size=0.9):
        """Splits the tweet data into training and test sets.

        Args:
            tweets (list): List of tweets to split.
            train_size (float): Proportion of data to use for training (default: 0.9).

        Returns:
            Tuple: Lists of training and test tweets.
        """
        np.random.seed(self.seed)
        n = len(tweets)
        train_size = int(n * train_size)
        lst = list(range(n))
        np.random.shuffle(lst)
        train_tweets = [tweets[i] for i in lst[:train_size]]
        test_tweets = [tweets[i] for i in lst[train_size:]]
        return train_tweets, test_tweets

    def _encode_and_filter_tweets(self, train_tweets, test_tweets):
        """Encodes and filters tweets based on minimum token count.

        Args:
            train_tweets (list): List of training tweets.
            test_tweets (list): List of test tweets.

        Returns:
            Tuple: Filtered encoded training and test tweets.
        """
        train_encoded = self.tokenizer(train_tweets, padding=False, truncation=False)['input_ids']
        test_encoded = self.tokenizer(test_tweets, padding=False, truncation=False)['input_ids']

        train_encoded_filtered = [tweet for tweet in train_encoded if len(tweet) >= self.min_token_count]
        test_encoded_filtered = [tweet for tweet in test_encoded if len(tweet) >= self.min_token_count]

        return train_encoded_filtered, test_encoded_filtered

    def _pad_sequences(self, sequences):
        """Pads or truncates sequences to a fixed length using TensorFlow.

        Args:
            sequences (list): List of encoded tweet sequences.

        Returns:
            numpy.ndarray: Padded sequences of shape (num_samples, max_seq_len).
        """
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=self.max_seq_len,
            padding='post',
            truncating='post',
            value=0
        )

    def get_train_test_data(self):
        """Returns the preprocessed training and test data.

        Returns:
            Tuple: Padded training and test sets for original and fake tweets.
        """
        return (self.trainList_orig_encoded_padded, self.testList_orig_encoded_padded, 
                self.trainList_fake_encoded_padded, self.testList_fake_encoded_padded)

class TransformerEncoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, max_seq_len, dropout_rate=0.1, output_size=128):
        """Initializes the TransformerEncoder model.

        Args:
            num_layers (int): Number of transformer encoder layers.
            d_model (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dff (int): Feed forward network dimension.
            input_vocab_size (int): Size of the input vocabulary.
            max_seq_len (int): Maximum sequence length.
            dropout_rate (float): Dropout rate for regularization (default: 0.1).
            output_size (int): Output embedding size (default: 128).
        """
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(max_seq_len, self.d_model)

        self.enc_layers = [self.encoder_layer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.final_layer = Lambda(lambda x: K.l2_normalize(x, axis=-1))

    def call(self, inputs, training=False, mask=None):
        """Forward pass for the TransformerEncoder.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, seq_len).
            training (bool): Whether the model is in training mode (default: False).
            mask (tf.Tensor): Padding mask for attention (default: None).

        Returns:
            tf.Tensor: Encoded output tensor of shape (batch_size, seq_len, d_model).
        """
        seq_len = tf.shape(inputs)[1]
        if mask is None:
            mask = self.create_padding_mask(inputs)

        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i]([x, mask], training=training)

        x = self.final_layer(x)
        return x

    def encoder_layer(self, d_model, num_heads, dff, dropout_rate):
        """Creates a single transformer encoder layer.

        Args:
            d_model (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dff (int): Feed forward network dimension.
            dropout_rate (float): Dropout rate.

        Returns:
            tf.keras.Model: A transformer encoder layer model.
        """
        inputs = tf.keras.Input(shape=(None, d_model))
        padding_mask = tf.keras.Input(shape=(1, 1, None))

        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model)(inputs, inputs, inputs)
        attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)

        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        ffn_output = ffn(attention_output)
        ffn_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)

        return tf.keras.Model(inputs=[inputs, padding_mask], outputs=ffn_output)

    def create_padding_mask(self, seq):
        """Creates a padding mask for the input sequence.

        Args:
            seq (tf.Tensor): Input sequence tensor of shape (batch_size, seq_len).

        Returns:
            tf.Tensor: Padding mask of shape (batch_size, 1, 1, seq_len).
        """
        last_non_zero_index = tf.reduce_sum(tf.cast(tf.math.not_equal(seq, 0), tf.int32), axis=-1) - 1
        mask = tf.sequence_mask(last_non_zero_index, maxlen=tf.shape(seq)[-1], dtype=tf.float32)
        mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)
        return mask

    def positional_encoding(self, max_seq_len, d_model):
        """Generates positional encodings for the input sequence.

        Args:
            max_seq_len (int): Maximum sequence length.
            d_model (int): Embedding dimension.

        Returns:
            tf.Tensor: Positional encoding tensor of shape (1, max_seq_len, d_model).
        """
        angle_rads = self.get_angles(
            tf.range(max_seq_len, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def get_angles(self, pos, i, d_model):
        """Calculates angles for positional encoding.

        Args:
            pos (tf.Tensor): Position indices.
            i (tf.Tensor): Dimension indices.
            d_model (int): Embedding dimension.

        Returns:
            tf.Tensor: Angle values for positional encoding.
        """
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        """Initializes the TripletLossLayer with a margin.

        Args:
            alpha (float): Margin for triplet loss.
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        """Computes the triplet loss for anchor, positive, and negative embeddings.

        Args:
            inputs (list): List of [anchor, positive, negative] embeddings.

        Returns:
            tf.Tensor: Triplet loss value.
        """
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor - positive), axis=(1, 2))
        n_dist = K.sum(K.square(anchor - negative), axis=(1, 2))
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0))

    def call(self, inputs):
        """Computes and adds the triplet loss during the forward pass.

        Args:
            inputs (list): List of [anchor, positive, negative] embeddings.

        Returns:
            tf.Tensor: Triplet loss value.
        """
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

def build_transformer():
    """Builds and returns a TransformerEncoder model with specified hyperparameters.

    Returns:
        TransformerEncoder: Configured transformer encoder model.
    """
    transformer = TransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        max_seq_len=max_seq_len,
        dropout_rate=dropout_rate,
        output_size=output_size
    )
    return transformer

def build_model(input_shape, network, margin):
    """Builds the Siamese Neural Network model with triplet loss.

    Args:
        input_shape (tuple): Shape of the input sequences.
        network (tf.keras.Model): Transformer encoder network.
        margin (float): Margin for triplet loss.

    Returns:
        Tuple: Training model (with triplet loss) and evaluation model (single input).
    """
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input")

    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)

    loss_layer = TripletLossLayer(alpha=margin, name='triplet_loss_layer')([encoded_a, encoded_p, encoded_n])

    network_train = Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss_layer)
    network_eval = Model(inputs=anchor_input, outputs=encoded_a)

    return network_train, network_eval

def get_batch_random(batch_size, X_orig, X_fake):
    """Creates a batch of random anchor-positive-negative (APN) triplets.

    Args:
        batch_size (int): Number of triplets to generate.
        X_orig (numpy.ndarray): Original tweet dataset (train or test).
        X_fake (numpy.ndarray): Fake tweet dataset (train or test).

    Returns:
        list: List of 3 tensors [A, P, N] of shape (batch_size, max_seq_len).
    """
    w = max_seq_len
    triplets = [np.zeros((batch_size, w)) for _ in range(3)]

    for i in range(batch_size):
        anchor_class = np.random.randint(2)
        if anchor_class == 0:
            X = X_orig
            X_N = X_fake
        else:
            X = X_fake
            X_N = X_orig

        nb_sample_available_for_class_AP = len(X)
        [idx_A, idx_P] = np.random.choice(nb_sample_available_for_class_AP, size=2, replace=False)
        idx_N = np.random.randint(0, len(X_N))

        triplets[0][i, :] = X[idx_A]
        triplets[1][i, :] = X[idx_P]
        triplets[2][i, :] = X_N[idx_N]

    return triplets

def get_batch_semi_hard2(draw_batch_size, semihard_batch_size, network, margin, X_orig, X_fake):
    """Creates a batch of semi-hard APN triplets.

    Args:
        draw_batch_size (int): Number of initial randomly sampled triplets.
        semihard_batch_size (int): Number of semi-hard triplets to select.
        network (tf.keras.Model): Siamese neural network model for embeddings.
        margin (float): Margin value for triplet loss.
        X_orig (numpy.ndarray): Original tweet dataset.
        X_fake (numpy.ndarray): Fake tweet dataset.

    Returns:
        list: List of 3 tensors [A, P, N] of shape (semihard_batch_size, max_seq_len).
    """
    study_batch = get_batch_random(draw_batch_size, X_orig, X_fake)
    A = network.predict(study_batch[0], verbose=0)
    P = network.predict(study_batch[1], verbose=0)
    N = network.predict(study_batch[2], verbose=0)

    ap_distance = np.sum(np.square(A - P), axis=(1, 2))
    an_distance = np.sum(np.square(A - N), axis=(1, 2))

    semihard_triplets = [np.zeros((semihard_batch_size, len(study_batch[0][0]))) for _ in range(3)]
    studybatch_short = [[], [], []]
    j = 0
    for i in range(draw_batch_size):
        ap_dist = ap_distance[i]
        an_dist = an_distance[i]
        if an_dist < ap_dist + margin and an_dist > ap_dist and j < semihard_batch_size:
            semihard_triplets[0][j] = study_batch[0][i]
            semihard_triplets[1][j] = study_batch[1][i]
            semihard_triplets[2][j] = study_batch[2][i]
            j += 1
        else:
            studybatch_short[0].append(study_batch[0][i])
            studybatch_short[1].append(study_batch[1][i])
            studybatch_short[2].append(study_batch[2][i])

    i = 0
    while j < semihard_batch_size:
        semihard_triplets[0][j] = studybatch_short[0][i]
        semihard_triplets[1][j] = studybatch_short[1][i]
        semihard_triplets[2][j] = studybatch_short[2][i]
        i += 1
        j += 1

    return semihard_triplets

def compute_dist(a, b):
    """Computes the squared L2 distance between two embeddings.

    Args:
        a (numpy.ndarray): First embedding.
        b (numpy.ndarray): Second embedding.

    Returns:
        float: Squared L2 distance between the embeddings.
    """
    return np.sum(np.square(a - b))

def compute_probs(network, X, Y):
    """Computes pairwise distances between embeddings for evaluation.

    Args:
        network (tf.keras.Model): Current neural network to compute embeddings.
        X (tf.Tensor): Input tensor of shape (num_samples, max_seq_len).
        Y (tf.Tensor): True labels of shape (num_samples,).

    Returns:
        Tuple: Arrays of pairwise distances and corresponding labels.
    """
    m = X.shape[0]
    nbevaluation = int(m * (m - 1) / 2)
    probs = np.zeros((nbevaluation))
    y = np.zeros((nbevaluation))

    embeddings = network(X)
    k = 0
    for i in range(m):
        for j in range(i + 1, m):
            probs[k] = -compute_dist(embeddings[i, :], embeddings[j, :])
            y[k] = 1 if Y[i] == Y[j] else 0
            k += 1
    return probs, y

def compute_metrics(probs, yprobs):
    """Computes ROC curve metrics and AUC score.

    Args:
        probs (numpy.ndarray): Array of pairwise distances.
        yprobs (numpy.ndarray): Array of true pairwise labels.

    Returns:
        Tuple: False positive rates, true positive rates, thresholds, and AUC score.
    """
    auc = roc_auc_score(yprobs, probs)
    fpr, tpr, thresholds = roc_curve(yprobs, probs)
    return fpr, tpr, thresholds, auc

def batch_accuracy(network_eval, triplets):
    """Computes the accuracy of triplets based on embedding distances.

    Args:
        network_eval (tf.keras.Model): Siamese network evaluation model (single input).
        triplets (list): List of 3 tensors [A, P, N], each of shape (batch_size, max_seq_len).

    Returns:
        float: Fraction of triplets where dist(A, P) < dist(A, N).
    """
    if not isinstance(triplets, (list, tuple)) or len(triplets) != 3:
        raise ValueError("triplets must be a list of 3 tensors")
    A_batch, P_batch, N_batch = triplets
    if A_batch.shape != P_batch.shape or A_batch.shape != N_batch.shape:
        raise ValueError("All triplets must have the same shape")
    
    all_inputs = tf.concat([A_batch, P_batch, N_batch], axis=0)
    all_embeddings = network_eval.predict(all_inputs, verbose=0)
    emb_a, emb_p, emb_n = tf.split(all_embeddings, 3, axis=0)
    
    p_dist = tf.reduce_sum(tf.square(emb_a - emb_p), axis=(1, 2))
    n_dist = tf.reduce_sum(tf.square(emb_a - emb_n), axis=(1, 2))
    
    accuracy = tf.reduce_mean(tf.cast(p_dist < n_dist, tf.float32))
    return accuracy.numpy()

def main():
    """Main function to process the tweet dataset, train the Siamese Neural Network, and save results.

    Parses command-line arguments, initializes the dataset, builds and trains the model,
    and saves the model and training history.
    """
    parser = argparse.ArgumentParser(description="Run Siamese Neural Network for tweet similarity")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument('--seed', type=int, required=True, help="Random seed for reproducibility")
    args = parser.parse_args()

    dataset_path = args.dataset
    seed = args.seed

    logging.info(f"Starting main function with dataset: {dataset_path}, seed: {seed}")
    set_seed(seed)

    try:
        dataset = TweetDataset(
            file_path=dataset_path,
            seed=seed,
            tokenizer_name="gpt2",
            min_token_count=min_token_count,
            max_seq_len=max_seq_len
        )
    except Exception as e:
        logging.error(f"Failed to initialize dataset: {e}")
        sys.exit(1)

    try:
        train_orig, test_orig, train_fake, test_fake = dataset.get_train_test_data()
        logging.info("Successfully retrieved preprocessed data.")
    except Exception as e:
        logging.error(f"Failed to retrieve preprocessed data: {e}")
        sys.exit(1)

    input_shape = (max_seq_len,)
    logging.info("Building Transformer model...")
    network = build_transformer()

    logging.info("Building Siamese Neural Network model...")
    network_train, network_eval = build_model(input_shape, network, margin)

    logging.info("Compiling model...")
    network_train.compile(optimizer=Adam(learning_rate=learning_rate), loss=None)

    logging.info("=== Model Summary ===")
    network_train.summary()
    
    emb = network_eval(train_orig[:1])
    logging.info(f"One‑batch embedding shape: {emb.shape}")
    
    logging.info("Starting training process!")
    logging.info("-------------------------------------")
    n_iter = 600
    evaluate_every = 10
    n_iteration = 0
    val_batch_size = 600
    t_start = time.time() 
    
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    last_train_acc = 0.0
    last_val_loss = 0.0
    last_val_acc = 0.0

    for i in range(1, n_iter + 1):
        triplets_train = get_batch_semi_hard2(
            draw_batch_size=1500,
            semihard_batch_size=200,
            network=network_eval,
            margin=margin,
            X_orig=train_orig,
            X_fake=train_fake
        )
        
        train_loss = network_train.train_on_batch(triplets_train, None)
        train_acc = batch_accuracy(network_eval, triplets_train)
        
        n_iteration += 1
        if i % evaluate_every == 0:
            triplets_val = get_batch_random(val_batch_size, test_orig, test_fake)
            val_loss = network_train.test_on_batch(triplets_val, None)
            val_acc = batch_accuracy(network_eval, triplets_val)
            
            train_loss_history.append(float(train_loss))
            train_acc_history.append(float(train_acc))
            val_loss_history.append(float(val_loss))
            val_acc_history.append(float(val_acc))

            logging.info(
                f"Iter {i}/{n_iter} — "
                f"Time: {(time.time() - t_start) / 60.0:.1f} mins — "
                f"Train Loss: {train_loss:.4f} — "
                f"Train Acc: {train_acc:.4f} — "
                f"Val Loss: {val_loss:.4f} — "
                f"Val Acc: {val_acc:.4f} — "
            )
            
            last_train_acc = float(train_acc)
            last_val_loss = float(val_loss)
            last_val_acc = float(val_acc)
    
    MODEL_PATH = Path.cwd()
    MODEL_NAME = f"snn_v2_run_{seed}.keras"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    
    network_eval.save(MODEL_SAVE_PATH)
    logging.info(f"[run {seed}] saved full inference model at {MODEL_SAVE_PATH}")
    
    history = {
        "train_loss_history": train_loss_history,
        "train_acc_history": train_acc_history,
        "val_loss_history": val_loss_history,
        "val_acc_history": val_acc_history
    }
    with open(f"training_runhistory_v2_{seed}.json", "w") as f:
        json.dump(history, f, indent=4)
    logging.info(f"File saved as train_runhistory_{seed}.json")

if __name__ == "__main__":
    main()