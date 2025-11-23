# AI Chatbot System

A complete, production-ready AI Chatbot system built with Python, NLTK, TensorFlow/Keras, and Flask. The chatbot supports intent classification, entity recognition, and response generation using deep learning.

## 🚀 Features

- **Intent Classification**: Deep learning model using TensorFlow/Keras for accurate intent detection
- **Entity Recognition**: NLTK-based NER to extract locations, dates, names, and keywords
- **Response Generation**: Context-aware responses with entity insertion
- **Flask API**: RESTful API endpoint for chat interactions
- **Modern UI**: Beautiful, responsive web interface with real-time chat
- **Confidence Scores**: Returns confidence levels for each prediction
- **Unknown Input Handling**: Gracefully handles unrecognized inputs

## 📁 Project Structure

```
chatbot/
├── data/
│   └── intents.json          # Training data with intents, patterns, and responses
├── model/                    # Trained model files (generated after training)
│   ├── chatbot_model.h5     # Trained neural network model
│   ├── tokenizer.pkl        # Word tokenizer
│   └── label_encoder.pkl    # Label encoder for intents
├── templates/
│   └── index.html           # Frontend HTML template
├── static/
│   ├── style.css            # CSS styling
│   └── script.js            # Frontend JavaScript
├── training.py              # Model training script
├── chatbot.py               # Main chatbot module
├── ner.py                   # Entity recognition module
├── app.py                   # Flask application
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 🔧 Installation

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### 2. Install Dependencies

```bash
cd chatbot
pip install -r requirements.txt
```

### 3. Download NLTK Data

The scripts will automatically download required NLTK data on first run. If you encounter issues, manually download:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

## 🎯 Usage

### Step 1: Train the Model

First, train the chatbot model using the training script:

```bash
python training.py
```

This will:
- Load intents from `data/intents.json`
- Preprocess the data (tokenization, lemmatization, bag-of-words)
- Train a neural network model
- Save the model files to `model/` directory

**Expected Output:**
```
============================================================
AI CHATBOT TRAINING
============================================================
Loading intents data...
Loaded 50 documents
Found 150 unique words
Found 10 classes

Creating training data...
Training data shape: (50, 150)
Labels shape: (50, 10)

Train set: 40 samples
Test set: 10 samples

Building neural network model...
Training model...
...
Test Loss: 0.1234
Test Accuracy: 0.9500

✓ Model saved to model/chatbot_model.h5
✓ Tokenizer saved to model/tokenizer.pkl
✓ Label encoder saved to model/label_encoder.pkl
```

### Step 2: Run the Flask Application

Start the Flask server:

```bash
python app.py
```

The server will start at `http://127.0.0.1:5000`

### Step 3: Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

You'll see a beautiful chat interface where you can interact with the chatbot.

## 📡 API Usage

### Chat Endpoint

**POST** `/chat`

Send a message to the chatbot and get a response.

**Request:**
```json
{
  "message": "Hello, how are you?"
}
```

**Response:**
```json
{
  "response": "Hello! How can I help you today?",
  "intent": "greeting",
  "confidence": 0.95,
  "confidence_scores": {
    "greeting": 0.95,
    "goodbye": 0.02,
    "thanks": 0.01,
    ...
  },
  "entities": {
    "locations": [],
    "dates": [],
    "names": [],
    "keywords": []
  },
  "extracted_location": null,
  "extracted_date": null,
  "extracted_name": null,
  "error": null
}
```

**Example using curl:**
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Book a flight to Bangalore"}'
```

### Health Check Endpoint

**GET** `/health`

Check if the chatbot is ready.

**Response:**
```json
{
  "status": "healthy",
  "chatbot_ready": true
}
```

## 🧪 Testing

### Test the Chatbot Module

```bash
python chatbot.py
```

### Test Entity Recognition

```bash
python ner.py
```

## 🎨 Customization

### Adding New Intents

Edit `data/intents.json` to add new intents:

```json
{
  "tag": "new_intent",
  "patterns": [
    "pattern 1",
    "pattern 2",
    "pattern 3"
  ],
  "responses": [
    "Response 1",
    "Response 2"
  ]
}
```

After adding new intents, retrain the model:
```bash
python training.py
```

### Modifying the Model Architecture

Edit `training.py` to change the neural network architecture:

```python
def build_model(input_shape, output_shape):
    model = Sequential([
        Dense(256, input_shape=(input_shape,), activation='relu'),  # Increase neurons
        Dropout(0.5),
        Dense(128, activation='relu'),  # Add more layers
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(output_shape, activation='softmax')
    ])
    ...
```

### Adjusting Training Parameters

In `training.py`, modify:
- `epochs`: Number of training epochs (default: 200)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 0.001)
- `test_size`: Test split ratio (default: 0.2)

## 🔍 How It Works

### 1. Text Preprocessing
- Converts text to lowercase
- Removes special characters
- Tokenizes the text
- Lemmatizes words
- Removes stopwords
- Creates bag-of-words vector

### 2. Intent Classification
- Uses trained neural network to predict intent
- Returns confidence scores for all intents
- Selects the intent with highest confidence

### 3. Entity Recognition
- Extracts named entities (locations, dates, names)
- Uses NLTK's NER and pattern matching
- Identifies keywords in context

### 4. Response Generation
- Matches intent to response templates
- Randomly selects a response from available options
- Inserts detected entities into response
- Returns formatted response with metadata

## 🛠️ Technical Details

### Model Architecture
- **Input Layer**: Bag-of-words vector (vocabulary size)
- **Hidden Layer 1**: 128 neurons, ReLU activation, 50% dropout
- **Hidden Layer 2**: 64 neurons, ReLU activation, 50% dropout
- **Output Layer**: Softmax activation (number of intents)
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical crossentropy

### Preprocessing Pipeline
1. Lowercase conversion
2. Special character removal
3. Tokenization (NLTK)
4. Lemmatization (WordNet)
5. Stopword removal
6. Bag-of-words vectorization

### Entity Recognition
- Uses NLTK's Named Entity Chunker
- Pattern-based extraction for locations
- Keyword matching for dates
- Capitalized word detection

## 📊 Performance

- **Training Accuracy**: Typically 90-95%
- **Test Accuracy**: Typically 85-95%
- **Response Time**: < 100ms (without GPU)
- **Response Time**: < 50ms (with GPU)

## 🐛 Troubleshooting

### Issue: Model files not found
**Solution**: Run `python training.py` first to generate model files.

### Issue: NLTK data not found
**Solution**: The script will auto-download, but you can manually download:
```python
import nltk
nltk.download('all')
```

### Issue: Low accuracy
**Solution**: 
- Add more training patterns in `intents.json`
- Increase training epochs
- Adjust model architecture
- Add more diverse examples

### Issue: Import errors
**Solution**: Make sure you're in the chatbot directory and all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 🚀 GPU Support

If you have a GPU available, TensorFlow will automatically use it. To verify:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

For better GPU performance, install:
```bash
pip install tensorflow-gpu
```

## 📝 License

This project is open source and available for educational and commercial use.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning framework
- NLTK for natural language processing
- Flask for web framework
- All contributors and users

## 📧 Support

For issues or questions, please open an issue on the repository.

---

**Happy Chatting! 🎉**

#   a i - c h a t b o t  
 