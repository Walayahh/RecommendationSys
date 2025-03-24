from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Retrieve selected perfumes (as comma‐separated string) and gender from the form.
    perfumes = request.form.get('perfumes', '')
    gender = request.form.get('gender', '')
    # Split the list of perfumes (trimming extra whitespace)
    perfume_list = [p.strip() for p in perfumes.split(',') if p.strip()]
    return render_template('recommendations.html', perfumes=perfume_list, gender=gender)

if __name__ == '__main__':
    app.run(debug=True)
