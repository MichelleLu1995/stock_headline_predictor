

from flask import Flask, render_template, request
import pandas_datareader.data as web
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/')
def main():
	df = pd.read_csv('constituents_csv.csv')
	companies = df['Name'].values
	companies = companies.tolist()
	return render_template("main.html", company_arr=companies)

@app.route('/company', methods=['GET', 'POST'])
def searchCompany():
	company = request.form['company']
	print(company)
	return render_template("search.html", company=company)


if __name__ == "__main__":
	app.run(debug=True)

