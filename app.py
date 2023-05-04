# Imports:
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

# import update function from local dir
from update import update_model

# import HashingVectorizer from local dir
from vectorizer import vect

# Creates the flask app
app = Flask(__name__)

# Prepares the classifier:
# Gets the current directory
cur_dir = os.path.dirname(__file__)
# Database path
db = os.path.join(cur_dir, "reviews.sqlite")
# Creates a classifier from the pickled SGD Classifier.
clf = pickle.load(open(os.path.join(cur_dir, "pkl_objects", "classifier.pkl"), "rb"))
# If the program is run directly
if __name__ == "__main__":
    # Update the model/data using the update file.
    clf = update_model(db_path=db, model=clf, batch_size=1000)


# Classifies an input as either positive or negative.
def classify(document):
    # Label options
    label = {0: "negative", 1: "positive"}
    # Gets the X (actual review) data from the document
    X = vect.transform([document])
    # Uses the classifier to predict the corresponding y value.
    y = clf.predict(X)[0]
    # Determines the probability of the prediction being correct.
    proba = np.max(clf.predict_proba(X))
    # Returns the prediction and its probability.
    return label[y], proba


# Function to train the data
def train(document, y):
    # Gets the X (review) data from the document
    X = vect.transform([document])
    # Fits the classifier
    clf.partial_fit(X, [y])


# Enters the new data into the reviews sql file.
def sqlite_entry(path, document, y):
    # Connects and gets the cursor
    conn = sqlite3.connect(path)
    c = conn.cursor()
    # Enters the review and sentiment data into the database
    c.execute(
        "INSERT INTO review_db (review, sentiment)" " VALUES (?, ?)", (document, y)
    )
    conn.commit()
    conn.close()


# Website building:
# Review form class:
class ReviewForm(Form):
    # Formats the icecream review
    creamyreview = TextAreaField(
        "", [validators.DataRequired(), validators.length(min=15)]
    )


@app.route("/")
def index():
    # returns the formatted page.
    return render_template("reviewform.html")


@app.route("/modal")
def modal():
    form = ReviewForm(request.form)
    # returns the formatted form.
    return render_template("modal.html", form=form)


# Gets the results of the form
@app.route("/results", methods=["POST"])
def results():
    # Creates a form using the review form class
    form = ReviewForm(request.form)
    if request.method == "POST" and form.validate():
        # Gets the review
        review = request.form["creamyreview"]
        # Classifies the review and gets its probability of being correct
        y, proba = classify(review)
        # Returns the formatted results
        return render_template(
            "results.html",
            content=review,
            prediction=y,
            probability=round(proba * 100, 2),
        )
    return render_template("reviewform.html", form=form)


# Gets the user's feedback on the classification response.
@app.route("/thanks.html", methods=["POST"])
def thanks():
    # Feedback button
    feedback = request.form["feedback_button"]
    # Review
    review = request.form["review"]
    # Prediction
    prediction = request.form["prediction"]

    # Creates the labels
    inv_label = {"negative": 0, "positive": 1}
    # Gets the y label using the prediction.
    y = inv_label[prediction]
    # If the feedback indicates that the prediction was wrong,
    if feedback == "Incorrect":
        # y must be the other class.
        y = int(not (y))
    # Trains with the review and the y class label.
    train(review, y)
    # Enters the data into the database.
    sqlite_entry(db, review, y)
    # Returns the thankyou template.
    return render_template("thanks.html")


# If the file is run directly:
if __name__ == "__main__":
    # app.run(host='0.0.0.0', debug=True)
    # Runs the app.
    app.run()
