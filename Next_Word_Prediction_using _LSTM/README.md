# LSTM Next-Word Prediction

A small text-generation project that predicts the next word from a user-provided prompt. It uses a pretrained Keras LSTM model and a Streamlit interface, with quotes as the training text source.

## Features

- Enter a prompt in a simple Streamlit web interface.
- Predict one next word using the saved LSTM model.
- Generate additional words from a seed prompt in the Python script.
- Includes the quote dataset and the tokenizer/model artifacts needed by the app.

## Project files

- `app.py` — Streamlit interface for interactive next-word prediction.
- `main.py` — data preprocessing, model setup, and text-generation code.
- `qoute_dataset .csv` — quote text and author dataset used by the project.
- `lstm_model (1).h5` — saved Keras model.
- `tokenizer.pkl` — fitted text tokenizer used to encode prompts.
- `max_len.pkl` — saved sequence-length setting used by the app.

## Requirements

- Python 3.9 or newer
- TensorFlow
- Streamlit
- NumPy
- pandas
- seaborn
- Matplotlib

Install the required packages in your Python environment:

```bash
python -m pip install streamlit tensorflow numpy pandas seaborn matplotlib
```

## Run the app

From this project directory, start Streamlit:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, enter a prompt, and select **Predict Next Word**. The saved model and tokenizer files must remain in this directory for the app to load them.

## Notes

The model predicts from the vocabulary and patterns learned from the supplied quote dataset; its output is a suggestion, not a guaranteed continuation. The dataset filename intentionally retains its original spelling (`qoute_dataset .csv`).
