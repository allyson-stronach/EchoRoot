from flask import Flask, render_template, redirect, request, session, flash
import model

app = Flask(__name__) 
app.secret_key = "7mghJqRp^tU1tE-me^A8TTPF]$Rr$;~gQYmM<8Zq{b7(f`BQ79bC.-/rB07T"

@app.route("/")
def index():
	ad_number = '1,000,000'
	att_number = '3,500,000'
	return render_template('index.html', ad_number=ad_number, att_number=att_number)

@app.route('/analyze', methods = ['POST'])
def signup():
    email = request.form['adform']
    print("The text in the ad you submitted is '" + new_ad + "'")
    return redirect('/')

if __name__== "__main__":
    app.run(debug = True)