# Text Sentiment Analysis Web App

A web-based AI application that analyzes text input and predicts the sentiment of the text as Positive, Neutral, or Negative. Built using Streamlit and Hugging Face Transformers.

## Features

* Real-time sentiment analysis for any text input.
* Multi-paragraph support.
* Displays sentiment category and confidence score.
* User-friendly Streamlit interface.
* Easy deployment to Streamlit Cloud.

## Technologies Used

* Python 3.10+
* Streamlit
* Transformers (Hugging Face)
* PyTorch
* tf-keras (for Transformers compatibility)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. Create a virtual environment (optional):

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the app locally:

```bash
streamlit run app.py
```

* Enter a sentence or paragraph.
* Click Analyze.
* See the predicted sentiment and confidence score.

## Deployment

* Push your project to GitHub.
* Go to [Streamlit Cloud](https://share.streamlit.io/) → New App → Connect your GitHub repo.
* Set `app.py` as the main file and deploy.

## Project Structure

```
text-sentiment-analysis/
├── app.py
├── requirements.txt
├── README.md
└── assets/  
```

## License

MIT License
