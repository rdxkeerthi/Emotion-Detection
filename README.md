### Text-based Emotion Detection Web App

This is a web-based application that detects emotions from text input using Natural Language Processing (NLP) and Machine Learning. The app processes the input text and predicts the associated emotion, displaying the result along with the prediction confidence.
### Features

   - **Real-time Emotion Detection:** Enter any text and receive instant emotion analysis.
   - **Emotion Classes:** The app can detect multiple emotions, including:
       - Anger 😠
       - Disgust 🤮
       - Fear 😨😱
       - Happiness 🤗
       - Joy 😂
       - Neutral 😐
       - Sadness 😔
       - Shame 😳
       - Surprise 😮
       
- **Confidence Score:** Provides a confidence score for each emotion prediction.

- **Visualization:** Displays a bar chart of the prediction probabilities for each emotion.

- **Usage Monitoring:** Track the number of page visits and emotion classification statistics.

### How It Works

1. The user types in a sentence or a block of text in the text area provided.
2. The system processes the input text and applies a pre-trained machine learning model to predict the emotion.
3. The detected emotion, along with an emoji, is displayed, and the confidence score for the prediction is shown.
4.  The user can see a probability distribution of all possible emotions for the given input.
5. All predictions and usage data are recorded for monitoring purposes.

### Tech Stack

- **Front-end:** Streamlit (Python)

- **Back-end:** Flask (Python)

- **Back-end:** scikit-learn (Machine Learning Model), joblib (Model Persistence)

- **Visualization:** Altair for charts, Plotly for pie charts

- **Database (Optional):** SQLite (for tracking usage metrics)

### Requirements

To run this application locally, you'll need:

- Python 3.8+
- The following Python libraries:

    ```bash 
    pip install streamlit pandas joblib scikit-learn altair plotly
    ```

### Running Locally

   - Clone the repository:

   ```bash 
   git clone https://github.com/rdxkeerthi/Emotion-Detection.git
```

```bash 
cd Emotion-Detection
```
### Install the dependencies:

```bash
pip install -r requirements.txt
```
### Download the pre-trained model:

- Place the model file ```emotion_classifier_pipe_lr.pkl``` in the models/ folder.

### Run the app:

```bash
streamlit run app.py
```
- The app should now be running on http://localhost:8501. You can visit this URL in your browser to access the application.

### Deployment on Streamlit Community Cloud

1. Push your project to a GitHub repository.
2. Go to **Streamlit Community Cloud**.
3. Click on New app and link your GitHub repository.
4. Choose the branch (usually ```main```) and select ```app.py``` as the main file.
5. Click **Deploy**.

Your app will be live, and you can share the link with others to detect emotions from their text!
### Model Training

The model used for emotion detection is trained using the following steps:

1. **Dataset:** The model is trained on an emotion-labeled text dataset (e.g., tweets, comments, etc.).
2. **Pipeline:** The pre-processing and model training pipeline includes:
3. **Text Vectorization:** Using ```TfidfVectorizer``` to convert text into feature vectors.
4. **Classifier:** Logistic Regression model trained on the processed text.
    Model Persistence: The trained model is saved as a ```.pkl``` file using ```joblib```.

To retrain the model, follow these steps:

1. Prepare the dataset.
2.  Modify and run the train_model.py script (not included here).
3. Save the new model as emotion_classifier_pipe_lr.pkl.

### Usage Monitoring

The app includes basic analytics for monitoring usage:

1. **Page Visits:** Track how often the app is visited.
2. **Emotion Predictions:** Track the predicted emotions over time.
3. **Visualization:** View usage metrics using bar charts and pie charts.

### Future Enhancements

1. **Audio-based Emotion Detection:** Extend the app to process speech/audio input to detect emotions.
2. **Multilingual Support:** Support for emotion detection in multiple languages.
3. **Real-time Feedback:** Provide users with immediate feedback on their emotional state based on input.
4. **Advanced Dashboard:** Build a dashboard for admins to monitor app usage, prediction statistics, and user engagement.

### License

This project is licensed under the MIT License.
### Contact

For any queries, feel free to reach out:

- [**Website**](https://rdxkeerthi.web.app)
- [**LinkedIn**](https://linkedin.com/in/rdxkeerthi)