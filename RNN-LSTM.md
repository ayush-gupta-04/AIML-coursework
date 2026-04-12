Absolutely — let’s do **RNN / LSTM** next 🧠✨

# 6) RNN / LSTM — Theory

These are deep learning models used for **sequence data**.

Sequence data means data where order matters, like:

* text
* speech
* time series
* stock prices
* sensor data

For your lab, the most common use is:

> **text classification**, especially sentiment analysis

---

## Why do we need RNNs?

Normal neural networks do not handle sequence order very well.

For example, in text:

* “movie was not good”
* “movie was good”

The order of words changes the meaning.

RNNs are designed to process data **one step at a time**, while remembering previous information.

---

## Core idea of RNN

An RNN reads a sequence step by step.

For a sentence, it reads:

* word 1
* word 2
* word 3
* word 4

At each step, it keeps a hidden memory of what it has seen so far.

So it can use both:

* current input
* previous context

---

## Example intuition

Sentence:

> “The movie was not great”

If the model only looks at “great”, it may think the sentence is positive.

But if it remembers “not”, it can understand the real meaning is negative.

That memory part is what RNNs are for.

---

## Problem with basic RNNs

Basic RNNs have difficulty remembering long-term information.

This is called the **vanishing gradient problem**.

In simple words:

* important earlier words may get forgotten
* long sentences become hard to understand

That is why LSTM was introduced.

---

## LSTM

**LSTM** stands for **Long Short-Term Memory**.

It is a special type of RNN that is better at remembering important information for longer periods.

LSTM uses gates to decide:

* what to remember
* what to forget
* what to output

---

## Main gates in LSTM

### 1. Forget gate

Decides what old information should be removed.

### 2. Input gate

Decides what new information should be stored.

### 3. Output gate

Decides what information should be sent forward as output.

This makes LSTM much better than basic RNN for long text.

---

## Where RNN / LSTM are used

* sentiment analysis
* text classification
* machine translation
* speech recognition
* stock prediction
* weather forecasting

---

## Difference between RNN and LSTM

### RNN

* simpler
* processes sequence step by step
* struggles with long dependencies

### LSTM

* more powerful
* remembers long-term context better
* commonly used in text tasks

For viva, you can say:

> “RNN is used for sequence data, while LSTM is an improved version of RNN that can remember long-term dependencies better.”

---

# 2) Practical Implementation in Python

We will do **sentiment classification** using the **IMDB movie review dataset**.

Why this dataset?

* very common
* easy to explain
* perfect for RNN/LSTM
* binary classification: positive or negative review

---

# 3) Clean Python Code

```python id="rnnlstm01"
# RNN / LSTM for sentiment classification using the IMDB dataset

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SimpleRNN, Dropout

# 1. Load the dataset
# num_words=10000 means we keep only the top 10,000 most frequent words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# 2. Pad the sequences so all reviews have the same length
max_len = 200
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# 3. Build the model
model = Sequential()

# Embedding layer converts word indices into dense vectors
model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_len))

# RNN / LSTM layer
model.add(LSTM(64))

# Optional dropout to reduce overfitting
model.add(Dropout(0.5))

# Final output layer for binary classification
model.add(Dense(1, activation='sigmoid'))

# 4. Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# 6. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# 7. Predict one review
sample_review = X_test[0].reshape(1, max_len)
prediction = model.predict(sample_review)

print("Predicted probability:", prediction[0][0])
print("Predicted class:", 1 if prediction[0][0] >= 0.5 else 0)
print("Actual class:", y_test[0])

# 8. Plot training accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

---

# 4) Explanation of the Code, End to End

## Importing libraries

```python id="imp1"
import numpy as np
import matplotlib.pyplot as plt
```

* `numpy` helps with array operations
* `matplotlib.pyplot` is used for plotting training graphs

```python id="imp2"
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SimpleRNN, Dropout
```

These are from Keras:

* `imdb` loads the sentiment dataset
* `pad_sequences` makes all reviews the same length
* `Sequential` builds the model layer by layer
* `Embedding` converts word IDs into vectors
* `LSTM` is the main sequence layer
* `Dense` gives final output
* `Dropout` helps reduce overfitting
* `SimpleRNN` is imported here, but in this code we use `LSTM`

---

## Loading the dataset

```python id="load1"
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
```

This loads the IMDB dataset.

### What is inside it?

* movie reviews already converted into numbers
* labels:

  * `0` = negative review
  * `1` = positive review

### `num_words=10000`

We keep only the most common 10,000 words.

Why?

* reduces complexity
* removes rare/unhelpful words
* keeps the model manageable

---

## Padding sequences

```python id="pad1"
max_len = 200
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
```

Reviews have different lengths.

Some reviews are short, some are long.

Neural networks need fixed-size inputs, so we make every review length 200.

### What does padding do?

* shorter reviews are padded with zeros
* longer reviews are cut off

### Why is this needed?

Because the model expects each input to have the same shape.

---

## Building the model

```python id="model1"
model = Sequential()
```

This creates a model where layers are added one after another.

---

## Embedding layer

```python id="emb1"
model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_len))
```

This is very important for text.

### What does it do?

It converts each word index into a dense vector of size 128.

### Why not use raw word numbers directly?

Because numbers like `5`, `20`, `300` do not carry meaning by themselves.

Embedding learns meaningful vector representations of words.

### Parameters

* `input_dim=10000` → vocabulary size
* `output_dim=128` → each word becomes a 128-dimensional vector
* `input_length=max_len` → each sequence has length 200

---

## LSTM layer

```python id="lstm1"
model.add(LSTM(64))
```

This is the heart of the model.

### What does it do?

It reads the review word by word and remembers important context.

### Why 64 units?

This is the number of memory units in the LSTM layer.

More units can learn more complex patterns, but also increase computation.

---

## Dropout

```python id="drop1"
model.add(Dropout(0.5))
```

This randomly turns off 50% of neurons during training.

### Why?

To reduce overfitting.

It forces the model not to rely too much on only a few neurons.

---

## Output layer

```python id="out1"
model.add(Dense(1, activation='sigmoid'))
```

Since this is a binary classification task:

* output `0` or `1`
* we use one neuron
* sigmoid converts output into probability between 0 and 1

### Interpretation

* near 0 → negative review
* near 1 → positive review

---

## Compile the model

```python id="comp1"
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Optimizer

`adam` updates weights efficiently.

### Loss

`binary_crossentropy` is used for binary classification.

### Metric

`accuracy` tells us how often predictions are correct.

---

## Train the model

```python id="train1"
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)
```

The model learns from training data.

### Parameters

* `epochs=5` → training passes through data 5 times
* `batch_size=64` → 64 samples per update
* `validation_split=0.2` → 20% of training data is used for validation

### Why validation?

To see whether the model is learning properly and not just memorizing.

---

## Evaluate the model

```python id="eval1"
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
```

This checks the performance on unseen test data.

### Why important?

Because training accuracy alone is not enough.

Test accuracy tells us how well the model generalizes.

---

## Predict one review

```python id="pred1"
sample_review = X_test[0].reshape(1, max_len)
prediction = model.predict(sample_review)
```

We take one review and reshape it for prediction.

### Why reshape to `(1, max_len)`?

Because the model expects a batch of inputs, even if there is only one review.

---

## Interpreting the prediction

```python id="pred2"
print("Predicted probability:", prediction[0][0])
print("Predicted class:", 1 if prediction[0][0] >= 0.5 else 0)
print("Actual class:", y_test[0])
```

### Predicted probability

This is the probability that the review is positive.

### Predicted class

* if probability ≥ 0.5 → positive
* else → negative

### Actual class

This is the true label from the dataset.

---

## Plot accuracy

```python id="plot1"
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

This shows how training and validation accuracy changed over epochs.

### Why useful?

It helps you see:

* whether the model is improving
* whether it is overfitting

---

# 5) What is happening conceptually?

Here is the flow in simple words:

1. Review text is converted into numbers
2. Numbers are padded to equal length
3. Embedding turns each word into a vector
4. LSTM reads the sequence step by step
5. Dense layer makes the final decision
6. Sigmoid gives probability of positive sentiment

---

# 6) Viva-ready explanation

You can say:

> “RNNs are designed for sequence data because they process inputs in order and keep memory of previous steps. LSTM is an improved RNN that can remember long-term dependencies using gates. In my implementation, I used the IMDB dataset for binary sentiment classification, where reviews were padded, embedded into vectors, processed by an LSTM layer, and then classified using a sigmoid output layer.”

That is a very solid answer.

---

# 7) Important viva questions and answers

## What is RNN?

A neural network designed for sequential data.

## Why is RNN used for text?

Because word order matters in text.

## What is the weakness of RNN?

It has difficulty remembering long-term dependencies.

## What is LSTM?

A special type of RNN that remembers important information for longer.

## Why use Embedding?

To convert word indices into meaningful dense vectors.

## Why use padding?

To make all sequences the same length.

## Why use sigmoid in the output layer?

Because this is a binary classification problem.

---

# 8) Mini recap

* RNN/LSTM are used for **sequence data**
* RNN processes data step by step
* LSTM is better at remembering long context
* We used **IMDB movie reviews** for sentiment analysis
* Main layers: **Embedding → LSTM → Dense(sigmoid)**
* Loss: **binary crossentropy**

If you want, I can next give you a **full revision sheet of all 6 topics together** in very short viva format.
