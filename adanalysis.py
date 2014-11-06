from flask import Flask, render_template, redirect, request, session, flash
import model

app = Flask(__name__) 
app.secret_key = "7mghJqRp^tU1tE-me^A8TTPF]$Rr$;~gQYmM<8Zq{b7(f`BQ79bC.-/rB07T"

@app.route("/")
def index():
    pass

if __name__== "__main__":
    app.run(debug = True)