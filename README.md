# AI Attention Visualizer


## Project Overview

AI Attention Visualizer is a beginner-friendly AI application that extracts text from study-note images using Tesseract OCR, converts the extracted words into numerical embeddings using Sentence Transformers, calculates attention scores using the scaled dot-product attention mechanism, and visualizes the attention given to each word.

The project demonstrates the basic workflow of combining OCR, text embeddings, and an attention mechanism in an interactive Streamlit application.

## Project Flow

```text
Study Note Image
       |
       v
   Tesseract OCR
       |
       v
  Extracted Text
       |
       v
  Extract Words
       |
       v
Sentence Embeddings
       |
       v
Scaled Dot-Product Attention
       |
       v
Word Attention Scores
       |
       v
   Visualization
```

## Features

* Upload study-note images in JPG, JPEG, or PNG format.
* Extract text from images using Tesseract OCR.
* Display the extracted text.
* Clean and filter the extracted words.
* Select up to 20 suitable words for visualization.
* Generate 384-dimensional word embeddings using Sentence Transformers.
* Generate Query, Key, and Value representations.
* Calculate scaled dot-product attention.
* Display attention scores using Streamlit progress bars.
* Identify the word with the highest calculated attention score.

## Technologies Used

* Python
* Streamlit
* Tesseract OCR
* PyTesseract
* Sentence Transformers
* NumPy
* Pillow

## Project Structure

```text
AI-Attention-Visualizer/
|
|-- app.py
|-- ocr.py
|-- embedding.py
|-- attention.py
|-- requirements.txt
|-- README.md
```
<img width="921" height="402" alt="image" src="https://github.com/user-attachments/assets/cf395d61-07fd-4d40-bb32-2058329551f8" />
<img width="909" height="389" alt="image" src="https://github.com/user-attachments/assets/03be6322-95e4-4acf-a8ea-a0f34acd93a4" />
<img width="901" height="368" alt="image" src="https://github.com/user-attachments/assets/ac63569b-8704-40e8-a6c2-2289f8b8e53a" />
<img width="859" height="368" alt="image" src="https://github.com/user-attachments/assets/d35aed6d-daa7-422b-a803-bb959cba03df" />
<img width="884" height="335" alt="image" src="https://github.com/user-attachments/assets/25232322-82df-436c-b5e7-38987f0dfa98" />
<img width="878" height="360" alt="image" src="https://github.com/user-attachments/assets/66c47abf-f25b-4422-9233-5c5d4cbb7ab6" />
<img width="914" height="408" alt="image" src="https://github.com/user-attachments/assets/90e50132-5af9-4468-96e8-1652274c63b1" />
<img width="747" height="401" alt="image" src="https://github.com/user-attachments/assets/defc4f9c-e138-4f2d-b3d2-a2f75e5ae25b" />
<img width="620" height="122" alt="image" src="https://github.com/user-attachments/assets/1923f3af-0584-4a22-b26f-5e748caf054c" />


## Installation

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install streamlit numpy pillow pytesseract sentence-transformers
```

## Tesseract OCR Setup

Tesseract OCR must be installed separately because it is an external OCR software and is not installed through Python's `pip`.

For Windows, the default installation path used by the project is:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

The Tesseract executable path is configured in `ocr.py`.

Make sure Tesseract OCR is installed before running the application.

## Running the Application

Open the project folder in VS Code and run:

```bash
python -m streamlit run app.py
```

The application will open in the browser.

The default local address is:

```text
http://localhost:8501
```

## How the Application Works

### 1. Image Upload

The user uploads a study-note image through the Streamlit file uploader.

### 2. OCR

The uploaded image is processed using Tesseract OCR through PyTesseract.

The extracted text is displayed in the application.

### 3. Word Extraction

The extracted text is divided into individual words.

Punctuation marks such as:

```text
. , ! ? ; : ( ) [ ] { }
```

are removed from the words.

Very short words containing two or fewer characters are also removed.

The application then selects a maximum of 20 words for attention visualization.

### 4. Word Embeddings

The selected words are converted into numerical representations using the Sentence Transformers model:

```text
all-MiniLM-L6-v2
```

The model produces 384-dimensional embeddings for each word.

### 5. Query, Key and Value

The word embeddings are transformed into Query, Key, and Value representations using randomly initialized projection matrices.

```text
Q = X × WQ
K = X × WK
V = X × WV
```

### 6. Scaled Dot-Product Attention

The attention scores are calculated using:

```text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V
```

The implementation first calculates:

```text
QKᵀ
```

and scales the result using:

```text
√dₖ
```

A softmax function is then applied to obtain normalized attention weights.

### 7. Word Attention Score

The attention weights are averaged to obtain one display score for each word.

These scores are normalized for visualization and displayed using Streamlit progress bars.

### 8. Highest Attention Word

The application identifies the word with the highest calculated attention score and displays it as the highest-attention word.

## Attention Implementation

The project uses a manually implemented scaled dot-product attention mechanism.

The Query, Key, and Value matrices are generated from the word embeddings:

```python
Q = X @ WQ
K = X @ WK
V = X @ WV
```

The attention score matrix is calculated as:

```python
scores = Q @ K.T
```

The scores are scaled:

```python
scaled_scores = scores / np.sqrt(d_k)
```

Finally, the softmax function produces the attention weights:

```python
attention_weights = softmax(scaled_scores)
```

The average attention value is then used as the word-level display score.

## Limitations

* OCR accuracy depends on the quality and clarity of the input image.
* Handwritten text may not be recognized accurately.
* Only the first 20 suitable words are visualized.
* The Sentence Transformer model generates embeddings independently for the selected words.
* Query, Key, and Value projection matrices are randomly initialized.
* The calculated highest-attention word should not be interpreted as a definitive measure of semantic importance.
* The project demonstrates the attention mechanism for educational purposes rather than extracting the actual internal attention maps of a pretrained Transformer model.

## Future Enhancements

* Add OCR bounding boxes to the original image.
* Highlight high-attention words directly on the uploaded image.
* Add support for Tamil and other languages.
* Display attention values in a heatmap.
* Add keyword extraction.
* Add study-note summarization.
* Add question-answering functionality.
* Allow users to download attention analysis reports.
* Use pretrained Transformer attention maps for more advanced visualization.

## Conclusion

AI Attention Visualizer demonstrates how OCR, word embeddings, and the scaled dot-product attention mechanism can be combined in an interactive application.

The system extracts text from study-note images, converts the extracted words into numerical embeddings, calculates attention scores, and presents the results through an easy-to-understand visualization.

The project provides a practical introduction to important concepts in Artificial Intelligence, Natural Language Processing, and Transformer-based attention mechanisms.

