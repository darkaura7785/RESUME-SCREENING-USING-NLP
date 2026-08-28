# RESUME-SCREENING-USING-NLP


##RUN resumescreening.py to create pkl files since the size was too large i couldnt upload it|
## Libraries Used

* **NumPy (`numpy`)**

  * Used for numerical operations and array manipulation.
  * Used in the pie chart for generating values with `np.linspace()`.

* **Pandas (`pandas`)**

  * Used for loading and handling the resume dataset.
  * Used for data manipulation, grouping, filtering, and analyzing resume categories.

* **Matplotlib (`matplotlib.pyplot`)**

  * Used for creating and displaying data visualizations.
  * Used to create bar charts and pie charts showing resume category distributions.

* **Seaborn (`seaborn`)**

  * Used for statistical data visualization.
  * Used to create the count plot showing the number of resumes in each category.

* **Regular Expressions (`re`)**

  * Used for cleaning and preprocessing resume text.
  * Removes URLs, hashtags, mentions, punctuation, special characters, non-ASCII characters, and extra spaces.

* **Scikit-learn (`sklearn`)**

  * Main machine learning library used in the project.
  * **LabelEncoder:** Converts text-based resume categories into numerical labels.
  * **TfidfVectorizer:** Converts resume text into numerical TF-IDF features.
  * **train_test_split:** Splits the dataset into training and testing sets.
  * **KNeighborsClassifier:** Used to train and evaluate a K-Nearest Neighbors classification model.
  * **SVC:** Used to train the Support Vector Classification model for resume category prediction.
  * **RandomForestClassifier:** Used to train and evaluate a Random Forest classification model.
  * **OneVsRestClassifier:** Enables the classifiers to handle multiple resume categories.
  * **accuracy_score:** Calculates model prediction accuracy.
  * **confusion_matrix:** Evaluates classification results using a confusion matrix.
  * **classification_report:** Provides precision, recall, F1-score, and support for each category.

* **Pickle (`pickle`)**

  * Used to save and load trained machine learning objects.
  * Saves the trained SVC model, TF-IDF vectorizer, and LabelEncoder as `.pkl` files.
  * Allows the trained model to be reused in the Streamlit application without retraining.

* **Streamlit (`streamlit`)**

  * Used to build the web-based user interface for the Resume Screening application.
  * Provides the resume upload functionality and displays the predicted category.

* **PyPDF2 (`PyPDF2`)**

  * Used to read PDF files.
  * Extracts text from uploaded PDF resumes.

* **python-docx (`docx`)**

  * Used to read Microsoft Word `.docx` files.
  * Extracts text from paragraphs of uploaded Word resumes.
