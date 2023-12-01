import io
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf


@st.cache(allow_output_mutation=True)
def load_model():
    models = []
    models.append(tf.keras.models.load_model('./models/fold0'))
    models.append(tf.keras.models.load_model('./models/fold1'))
    models.append(tf.keras.models.load_model('./models/fold2'))
    models.append(tf.keras.models.load_model('./models/fold3'))
    models.append(tf.keras.models.load_model('./models/fold4'))
    return models


def preprocess_image(img):
    # определеям начало и конец участка, который мы будем выделять из изображения
    start_pos = 200
    end_pos = 400
    len_data = -(start_pos - end_pos)
    x = np.empty(len_data)

    # поворачиваем изображение
    imgn = np.array(img.rotate(90, expand=True).convert(mode='L'))
    # принудительно делаем все пиксили или 255 или 0
    imgn[imgn != 255] = 0
    # цикл по выделенному участку изображения
    for i in range(start_pos, end_pos):
        # стартовая позиция для поиска границы областей на изображении (самый правый пиксель)
        k = 639
        while True:
            # условие того, что мы нашли границу областей текщий пиксель и следующий пиксель разного цвета
            if (int(imgar[k][i]) == 255) and (int(imgar[k - 1][i]) != 255):
                break
            # условие, что мы не нашли границу и дошли до левого края изображения
            if k == 1:
                break
            # если граница не найдена и мы не дошли до конца то смещаемся на 1 пиксель влево
            k -= 1
        x[i - start_pos] = k

    return x


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def model_predict(models, x):
    total_pred = np.zeros((x.shape[0], 1))
    j = 0
    for i in range(len(models)):
        pred = models[i].predict(np.expand_dims(x, -1))
        if np.max(np.round(pred)) == 0 or np.min(np.round(pred)) == 1:
            continue
        else:
            total_pred += pred
            j += 1

        return total_pred[-1] / j


def print_predictions(preds):
    if preds == 0:
        st.write("дефекта нет")
    else:
        st.write("дефект есть")


models = load_model()

st.title('Новая улучшенная классификации изображений в облаке Streamlit')
img = load_image()
result = st.button('Распознать изображение')
if result:
    x = preprocess_image(img)
    preds = model_predict(models, x)
    st.write('**Результаты распознавания:**')
    print_predictions(preds)
