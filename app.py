from flask import Flask, render_template, flash, redirect, url_for, request
from dotenv import load_dotenv
import os
from virality import get_virality_figure_html  # import our function from virality.py

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your_secret_key")

@app.route("/analyze", methods=["POST"])
def analyze():
    # your code here
    return render_template("home.html")

@app.route("/virality", methods=["GET"])
def virality():
    try:
        json_filename = "tweet_data.json"  # Ensure this file exists with the required structure
        graph_html = get_virality_figure_html(json_filename)
    except Exception as e:
        flash(f"Error generating virality graph: {e}", "danger")
        graph_html = ""
    return render_template("virality.html", graph_html=graph_html)

@app.route("/")
def home():
    return render_template("home.html", tweets=[], query="", graph_html="")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)