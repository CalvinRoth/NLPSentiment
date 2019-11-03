import tensorflow as tf
import pandas as pd
import numpy as np
import os
import re
import tensorflow_hub as hub


# #Ignore some warning messages but an optional compiler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

l_rate = 0.003
h_unit = [600, 200]
step = 3000

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["review"] = []
    data["score"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["review"].append(f.read())
            #This regex will from the name of a file like '250_1.txt' extract 1(numbers after _)
            data["score"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Process the dataset files.
def load_datasets(force_download=False):
    train_df = load_dataset(os.path.join("aclImdb","train"))
    test_df = load_dataset(os.path.join("aclImdb","test"))

    return train_df, test_df

tf.logging.set_verbosity(tf.logging.ERROR)
train_df, test_df = load_datasets()

# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.estimator.inputs.pandas_input_fn(
train_df, train_df["polarity"], num_epochs=None, shuffle=True)

# Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], shuffle=False)
# Prediction on the test set.
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
test_df, test_df["polarity"], shuffle=False)

embed_col = hub.text_embedding_column(
    key="review",
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")


classifier = tf.estimator.DNNClassifier(
    hidden_units= h_unit,
    feature_columns=[embed_col],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate= l_rate))



# Training for 1,000 steps means 128,000 training examples with the default
# batch size. This is roughly equivalent to 5 epochs since the training dataset
# contains 25,000 examples.
classifier.train(input_fn=train_input_fn, steps=step);

train_eval_result = classifier.evaluate(input_fn=predict_train_input_fn)
test_eval_result = classifier.evaluate(input_fn=predict_test_input_fn)

print ("{accuracy}".format(**train_eval_result))
print ("{accuracy}".format(**test_eval_result))
