from flask import Flask, render_template, redirect, url_for, request, flash
from models.data_ingetion import fetch_all_data
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your_secret_key")

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html", tweets=[], query="")

@app.route("/analyze", methods=["POST"])
def analyze():
    query = request.form.get("query")
    if not query:
        flash("No query provided", "danger")
        return redirect(url_for("home"))
    
    try:
        tweets = fetch_all_data(query)
    except Exception as e:
        flash(f"Error fetching data: {e}", "danger")
        tweets = []
    return render_template("home.html", tweets=tweets, query=query)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)