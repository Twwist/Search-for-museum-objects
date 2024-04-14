import pandas as pd
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from numpy import dot
from numpy.linalg import norm

# подходящие форматы фалойв
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
# грузим датасет
DATASET_PATH = "/home/adikon/python/emb1.pkl"
df = pd.read_pickle(DATASET_PATH)


# Проверка файла на нужный формат
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# косинусное расстояние
def cos_sim(v1, v2):
    return dot(v1, v2) / (norm(v1) * norm(v2))


# функция для показа топ 10 картинок
@app.route('/res/<name>', methods=['GET'])
def download_file(name):
    """
    короче сначала тут ищется сам эмбединг вместе с путем
    потом идет запись косинусных расстояний со всеми попарно
    дальше сортируем список по косиноснуму расстоянию
    берем топ 10 в список и делаем список путей
    картинки лежат в пути static/dataset
    дальше там рендерится темплейт но тут луччше мне скажите
    проще будет если я быренько там накидаю
    """
    for i in df:
        if i[1] == name:
            v1 = i[2]
            break
    res = []
    for i in df:
        if i[1] == name:
            continue
        else:
            v2 = i[2]
        res.append((i[0], i[1], cos_sim(v1, v2)))
    res.sort(key=lambda x: -x[-1])
    paths = [f'dataset/{i[0]}/{i[1]}' for i in res[:10]]
    return render_template('img.html', paths=paths)


# Это промежуточное окно между загрузкой файла и показом топ 10
@app.route('/<name>', methods=['GET', 'POST'])
def check_file(name):
    """
    просто показываем фотку
    можно добавить чтобы внизу писался класс
    тип вот фотка ее класс ГРафика
    и опиисание еще если успеете
    """
    if request.method == 'POST':
        return redirect(url_for('download_file', name=name))
    else:
        v1 = []
        for i in df:
            if i[1] == name:
                v1 = i
                break
        path = f'dataset/{v1[0]}/{v1[1]}'
        return f'''
    <!doctype html>
    <title>Класс</title>
    <link rel="stylesheet" 
                    href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/css/bootstrap.min.css" 
                    integrity="sha384-giJF6kkoqNQ00vy+HMDP7azOuL0xtbfIcaT9wjKHr8RbDVddVHyTfAAsrekwKmP1" 
                    crossorigin="anonymous">
    <h1 style="text-align: center;">Вот ваш файл</h1>
    <div>
    <div class="item active">
                <img src="{url_for('static', filename=path)}"
                     style="max-width:800;max-height:800px;display:block;margin-left:auto;margin-right:auto;">
    <div style="display: flex;
  justify-content: space-around;">
        <button onclick="document.location='/'" class="btn btn-primary">На главную</button>
    <form method=post enctype=multipart/form-data>
      <input class="btn btn-primary" type=submit value=Чекнуть style="display: block;
  margin-left: auto;
  margin-right: auto;
  margin-top: 10px;
  width: 100px;">
  </div>
    </form>
    </div>
    </html>
    '''

# просто загрузка файла
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Не могу прочитать файл')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Нет выбранного файла')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            return redirect(url_for('check_file', name=filename))
    return '''
    <!doctype html>
    <link rel="stylesheet" 
                    href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/css/bootstrap.min.css" 
                    integrity="sha384-giJF6kkoqNQ00vy+HMDP7azOuL0xtbfIcaT9wjKHr8RbDVddVHyTfAAsrekwKmP1" 
                    crossorigin="anonymous">
    <title>Загрузить новый файл</title>
    <div style="text-align:center; margin-top:100px">
    <h1>Загрузить новый файл</h1>
    <form method=post enctype=multipart/form-data>
      <input class="form-control-file" type=file name=file>
      <input class="btn btn-primary" type=submit value=Upload>
    </form>
    </div>
    </html>
    '''


app.run()
