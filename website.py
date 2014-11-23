from flask import Flask, render_template, redirect, request, session, flash
import model
import jinja2
import os

app = Flask(__name__) 
app.secret_key = "7mghJqRp^tU1tE-me^A8TTPF]$Rr$;~gQYmM<8Zq{b7(f`BQ79bC.-/rB07T"
app.jinja_env.undefined = jinja2.StrictUndefined

@app.route("/")
def index():
    """This is the 'cover' page of the EchoRoot site"""
    return render_template('index.html') 
    
@app.route("/adanalysis")
def adanalysis():
    """This is the big page containing the ad analysis buttons"""
    return render_template("adanalysis.html")

if __name__== "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)