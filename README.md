# Word-Level LSTM Model for Sentence Completion

## Overview
This project focuses on developing a word-level Long Short-Term Memory (LSTM) model for sentence completion. The model is trained on Shakespeare's plays to predict the next word in a sequence, leveraging the power of recurrent neural networks to understand contextual word relationships.

## Methodology
### 1. Data Preprocessing
- **Loading and Cleaning Data:** The text corpus is loaded, and stop words and punctuation are removed to enhance predictability.
- **Tokenization and Encoding:** Sentences are tokenized, and words are converted into numerical representations for efficient computation.

### 2. Model Development
- **LSTM Model Architecture:**
  - Three stacked LSTM layers to capture sequential dependencies.
  - Embedding layer for word representation.
  - Dense output layer to predict the next word.
- **Training the Model:** The processed dataset is used to train the model using categorical cross-entropy loss and the Adam optimizer.

### 3. Evaluation and Prediction
- The trained model is tested on unseen text sequences.
- Accuracy and loss metrics are analyzed to measure performance.
- The model generates word predictions based on input text sequences.

## Tools & Technologies
- **Libraries:** TensorFlow/Keras, NLTK, NumPy, Pandas
- **Dataset:** Shakespeare’s plays
- **Hardware:** GPU acceleration for efficient training

## Applications
- Text completion and auto-suggestions
- AI-powered creative writing assistance
- Style-based text generation in Shakespearean language

## Future Enhancements
- Extend the model to character-level predictions for finer control.
- Experiment with transformer-based architectures for improved performance.
- Train on a larger and more diverse dataset for better generalization.
