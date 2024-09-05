from crypt import methods

from flask import Flask, render_template,request, redirect, url_for

from PIL import Image

import time
app = Flask(__name__)

@app.route('/')
#@app.route('/', methods=['GET', 'POST'])

def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'Kein Bild ausgewählt'

        file = request.files['image']

        if file.filename == '':
            return 'Keine Datei ausgewählt'

        # Bild mit PIL öffnen
        img = Image.open(file)

        output_KNN = 1

        if output_KNN == 0:
            return redirect(url_for('plant_healthy'))
        else:
            return redirect(url_for('plant_sick'))

        # Wenn die Methode GET ist, leite den Benutzer auf die Startseite um oder zeige eine Nachricht an
    return redirect(url_for('index'))

@app.route('/plant_sick')
def plant_sick():
    return render_template('plant_sick.html')



@app.route('/plant_healthy')
def plant_healthy():
    return render_template('plant_healthy.html')
@app.route('/run-script')

def run_script():
    from test import healing
    result = healing()
    return f"Script output: {result}"

if __name__ == '__main__':
    app.run(debug=True)


