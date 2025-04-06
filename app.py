from flask import Flask, render_template, redirect, url_for, request, flash
from dotenv import load_dotenv
from echochamber_ai import fetch_all_data
import os

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your_secret_key")

@app.route("/", methods=["GET"])
def home():
    # Render the homepage with no tweets or graph initially.
    return render_template("home.html", tweets=[], query="", graph_html="")

@app.route("/analyze", methods=["POST"])
def analyze():
    # Import fetch_all_data inside the route to avoid issues during app startup.
    from echochamber_ai import fetch_all_data

    query = request.form.get("query")
    if not query:
        flash("No query provided", "danger")
        return redirect(url_for("home"))
    
    try:
        tweets, fig = fetch_all_data(query)
        import plotly
        graph_html = fig.to_html(full_html=False)
    except Exception as e:
        flash(f"Error fetching data: {e}", "danger")
        tweets = []
        graph_html = ""
    
    return render_template("home.html", tweets=tweets, query=query, graph_html=graph_html)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)