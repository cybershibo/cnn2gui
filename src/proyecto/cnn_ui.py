"""Controles de CNN separados por arquitectura, entrenamiento y ejecución."""
import pandas as pd
import streamlit as st

from proyecto.compute_ui import device_selector
from proyecto.cnn import CATALOG, data_root, load_dataset, predict_cnn, train_cnn
from proyecto.activation_ui import input_signature, visualization_options, run_inference, render_trace


def render_architecture():
    st.subheader('Arquitectura convolucional · TensorFlow/Keras')
    st.write('Basada en el ejemplo: convolución → agrupación → capa densa → clasificación.')
    blocks = st.slider('Bloques convolución + agrupación', 1, 3, 2)
    filters = [st.selectbox(f'Filtros del bloque {i+1}', [8, 16, 32, 64, 128], index=min(2+i, 4)) for i in range(blocks)]
    st.caption('Filtros 3×3 con ReLU; agrupación 2×2. Tres bloques usan padding same para conservar dimensiones compatibles.')
    dense = st.slider('Neuronas de la capa densa final', 8, 256, 64, step=8)
    dropout = st.slider('Dropout: proporción de neuronas desactivadas al entrenar', 0.0, 0.7, 0.2, step=0.1)
    st.write(f'Entrada 28×28×1 → bloques {filters} → Dense({dense}) → Dropout({dropout:.1f}) → 10 clases')
    return dict(filters=filters, dense=dense, dropout=dropout)


def render_training(architecture):
    filters, dense, dropout = architecture['filters'], architecture['dense'], architecture['dropout']
    dataset = st.selectbox('Dataset público y gratuito', list(CATALOG),
                           format_func=lambda key: CATALOG[key]['name'])
    entry = CATALOG[dataset]
    st.write(entry['description'])
    st.markdown(f"[Origen y condiciones del dataset]({entry['source']})")
    st.caption('Ambos datasets son compatibles: imágenes 28×28 en gris y diez clases. Fashion-MNIST: licencia MIT. MNIST: consulta las condiciones en el origen.')
    st.write('Clases: ' + ', '.join(entry['classes']))
    if st.button('Descargar y explorar dataset'):
        try:
            with st.spinner('Descargando o verificando la copia local…'):
                (images, labels), _ = load_dataset(dataset)
            st.image([image for image in images[:10]], caption=[entry['classes'][int(i)] for i in labels[:10]], width=80)
            st.success('Dataset disponible en el volumen persistente de Docker.')
        except Exception as exc:
            st.error(f'No se pudo preparar el dataset. Comprueba la conexión y vuelve a intentar: {exc}')
    st.caption('Entrenar también descarga el dataset seleccionado si aún no está disponible. Las descargas no necesitan cuentas ni claves.')
    device = device_selector('tensorflow', 'cnn-train-device')
    with st.form('cnn_training'):
        samples = st.selectbox('Imágenes del conjunto oficial de entrenamiento', [1000, 5000, 10000, 60000], index=1)
        epochs = st.slider('Épocas de la CNN', 1, 100, 5)
        batch = st.selectbox('Tamaño de lote de la CNN', [16, 32, 64, 128], index=3)
        rate = st.selectbox('Tasa de aprendizaje de la CNN', [0.0001, 0.001, 0.01], index=1)
        seed = st.number_input('Semilla de la CNN', 0, 2147483647, 42)
        st.write('Se reserva 1/6 de las imágenes seleccionadas para validación. La prueba usa las 10 000 imágenes oficiales independientes.')
        st.caption('Adam, entropía cruzada y salida softmax. Parada temprana tras tres épocas sin mejora de validación, recuperando los mejores pesos. El tiempo depende del dispositivo seleccionado.')
        submitted = st.form_submit_button('Entrenar CNN y guardar', type='primary')
    if submitted:
        bar, chart = st.progress(0), st.empty()
        def update(history):
            bar.progress(len(history)/epochs)
            chart.line_chart(pd.DataFrame(history).set_index('época'))
        try:
            with st.spinner('Preparando datos y entrenando la CNN…'):
                run = train_cnn(dataset, filters, dense, dropout, epochs, batch, rate, int(seed), samples, update, device=device)
            st.session_state['last_cnn'] = run
            bar.progress(1.0)
            st.success('CNN guardada. Abre Ejecución para probarla.')
        except Exception as exc:
            st.error(f'No se completó el entrenamiento: {exc}')


def render_prediction(run, config):
    st.subheader('Probar la CNN guardada')
    st.write('Se convierte la imagen a gris, se redimensiona a 28×28 y se divide entre 255, como en el ejemplo.')
    sample = (data_root() / run / 'sample.png').read_bytes()
    upload = st.file_uploader('Imagen PNG o JPEG', type=['png', 'jpg', 'jpeg'], key='cnn_image')
    content = upload.getvalue() if upload else sample
    st.image(content, width=140)
    if not upload:
        st.caption(f"Ejemplo del test oficial. Clase real: {config['sample_expected']}.")
    device = device_selector('tensorflow', f'cnn-execute-{run}')
    enabled, delay = visualization_options(run)
    signature = input_signature(content+device.encode())
    if st.button('Clasificar imagen'):
        try:
            probabilities = run_inference(run, signature, lambda callback: predict_cnn(run, content, on_layer=callback, device=device), enabled, delay)
            st.success(f"Clase predicha: {config['classes'][int(probabilities.argmax())]}")
            st.bar_chart(pd.DataFrame({'Probabilidad': probabilities}, index=config['classes']))
            st.caption('La red solo conoce las diez clases de su dataset. Una fotografía diferente también recibirá una clase; eso no garantiza una predicción válida.')
        except Exception as exc:
            st.error(f'No se pudo clasificar la imagen: {exc}')
    if enabled:
        render_trace(run, signature)
