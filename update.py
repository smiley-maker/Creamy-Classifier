# Imports:
import pickle
import sqlite3
import numpy as np
import os

# import HashingVectorizer from local dir
from vectorizer import vect


# Function to update the model
def update_model(db_path, model, batch_size=1000):
    # Connects to the designated path and gets the cursor.
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT * from review_db")

    # Fetches the results.
    results = c.fetchmany(batch_size)
    while results:
        # Gets an array of the results
        data = np.array(results)
        # Separates into review and sentiment
        X = data[:, 0]
        y = data[:, 1].astype(int)

        # Class labels
        classes = np.array([0, 1])
        # Gets the training data from the review
        X_train = vect.transform(X)
        # Fits the model to the training data.
        model.partial_fit(X_train, y, classes=classes)
        # Gets the next result
        results = c.fetchmany(batch_size)

    # Closes the connection
    conn.close()
    # Returns the updated model
    return model


# Gets the current directory
cur_dir = os.path.dirname(__file__)

# Creates a new classifier from the pickled classifer.
clf = pickle.load(open(os.path.join(cur_dir, "pkl_objects", "classifier.pkl"), "rb"))
# Gets the database location
db = os.path.join(cur_dir, "reviews.sqlite")
# Updates the classifier
clf = update_model(db_path=db, model=clf, batch_size=1000)

# Uncomment the following lines if you are sure that
# you want to update your classifier.pkl file
# permanently.
# Adds the new data to the original database.
pickle.dump(
    clf, open(os.path.join(cur_dir, "pkl_objects", "classifier.pkl"), "wb"), protocol=4
)
