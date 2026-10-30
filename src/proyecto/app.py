"""Laboratorio con arquitectura, entrenamiento y ejecución independientes."""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

from proyecto.cnn_ui import render_architecture, render_prediction, render_training
from proyecto.training import load_run, predict, prepare, train


def saved_runs(kind=None):
    root = Path(os.environ.get('NN2GUI_DATA', 'data'))
    runs = []
    for directory in root.glob('*'):
        if (directory / 'history.json').exists():
            config, _, _ = load_run(directory.name)
            if kind is None or config.get('kind', 'mlp') == kind:
                runs.append(directory.name)
    return sorted(runs, reverse=True)


def run_label(run):
    config, _, _ = load_run(run)
    return f"{'CNN · ' + config['dataset'] if config.get('kind') == 'cnn' else 'MLP · ' + config['target']} · {run[:8]}"


def show_metrics(run):
    config, metrics, history = load_run(run)
    st.metric('Exactitud en prueba: ejemplos nuevos correctamente clasificados', f"{metrics['accuracy']:.1%}")
    st.caption('El test no ajusta pesos. Elige configuraciones con validación y reserva el test para evaluación final.')
    st.line_chart(pd.DataFrame(history).set_index('época'))
    st.write('Si baja la pérdida de entrenamiento y sube la de validación, puede haber sobreajuste.')
    st.write('Matriz de confusión: filas = clase real; columnas = clase predicha.')
    st.dataframe(pd.DataFrame(metrics['confusion'], index=config['classes'], columns=config['classes']))
    with st.expander('Configuración y particiones del experimento'):
        st.json(config)


def train_dense(architecture):
    st.subheader('Datos de aprendizaje')
    source = st.radio('Origen', ['Iris: ejemplo incluido', 'Mi archivo CSV'], horizontal=True)
    frame = None
    if source.startswith('Iris'):
        iris = load_iris(as_frame=True)
        frame = iris.data.copy()
        frame['especie'] = [iris.target_names[i] for i in iris.target]
        st.write('150 flores, cuatro medidas y tres especies. La red aprenderá a clasificar flores.')
    else:
        st.caption('CSV con encabezados, entradas numéricas y una columna de clase. Máximo 20 MB.')
        upload = st.file_uploader('Selecciona un CSV', type=['csv'])
        if upload:
            try:
                frame = pd.read_csv(upload)
            except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
                st.error(f'No se pudo leer el CSV: {exc}')
    if frame is None or not len(frame.columns):
        return
    st.dataframe(frame.head(10), use_container_width=True)
    target = st.selectbox('Columna que la red debe predecir', frame.columns, index=len(frame.columns)-1)
    try:
        _, _, columns, classes = prepare(frame, target)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.success(f'Datos compatibles: {len(frame)} ejemplos · {len(columns)} entradas · {len(classes)} clases')
    st.write(f"Arquitectura seleccionada: {len(columns)} → {' → '.join(map(str, architecture['hidden']))} → {len(classes)}")
    with st.form('dense_training'):
        epochs = st.slider('Épocas: recorridos por los datos de entrenamiento', 5, 300, 50)
        batch = st.selectbox('Ejemplos por lote', [8, 16, 32, 64, 128], index=2)
        rate = st.selectbox('Tasa de aprendizaje', [0.0001, 0.001, 0.01], index=1)
        seed = st.number_input('Semilla para repetir el experimento', 0, 2147483647, 42)
        st.write('Optimizador: Adam · Pérdida: entropía cruzada · Salida: una probabilidad por clase.')
        st.write('Separación: 60 % entrenamiento, 20 % validación, 20 % prueba. La normalización aprende solo del entrenamiento.')
        submitted = st.form_submit_button('Entrenar y guardar', type='primary')
    if submitted:
        bar, chart = st.progress(0), st.empty()
        def update(history):
            bar.progress(len(history)/epochs)
            chart.line_chart(pd.DataFrame(history).set_index('época'))
        try:
            with st.spinner('La red está aprendiendo…'):
                run = train(frame, target, architecture['hidden'], architecture['activation'], epochs, rate, batch, int(seed), update)
            st.session_state['last_run'] = run
            st.success('Modelo guardado. Abre Ejecución para probarlo.')
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(f'No se completó el entrenamiento: {exc}')


def predict_dense(run, config):
    st.subheader('Probar la red densa guardada')
    st.caption('Introduce las mismas medidas y unidades del dataset. Se usa la arquitectura y normalización del modelo guardado.')
    with st.form('prediction'):
        values = [st.number_input(name, value=float(mean), key=f'{run}-{i}')
                  for i, (name, mean) in enumerate(zip(config['columns'], config['mean']))]
        apply = st.form_submit_button('Predecir')
    if apply:
        try:
            probabilities = predict(run, values)
            st.success(f"Clase predicha: {config['classes'][int(probabilities.argmax())]}")
            st.bar_chart(pd.DataFrame({'Probabilidad': probabilities}, index=config['classes']))
            st.caption('Estas probabilidades no garantizan fiabilidad para entradas ajenas al dataset.')
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(str(exc))


st.set_page_config(page_title='Laboratorio de redes neuronales', page_icon='🧠', layout='wide')
st.title('Laboratorio de redes neuronales')
st.caption('Diseña la arquitectura · entrena con datos · ejecuta el modelo guardado')
st.info('Redes densas con PyTorch y convolucionales con TensorFlow. Ejecución en CPU dentro de Docker.')
with st.sidebar:
    st.header('Tu laboratorio')
    network = st.radio('Tipo de red para diseñar y entrenar', ['Densa (MLP)', 'Convolucional (CNN)'])
    st.caption('Ejecución permite abrir cualquier modelo guardado, independientemente del tipo seleccionado aquí.')
    st.caption('Los modelos y los datos se conservan en el volumen de Docker.')
kind = 'cnn' if network == 'Convolucional (CNN)' else 'mlp'
architecture_tab, training_tab, execution_tab = st.tabs(['1 · Arquitectura', '2 · Entrenamiento', '3 · Ejecución'])
with architecture_tab:
    st.header('Define la arquitectura')
    if kind == 'cnn':
        architecture = render_architecture()
    else:
        count = st.slider('Capas ocultas', 1, 4, 2)
        hidden = [st.slider(f'Neuronas en capa {i+1}', 4, 128, 32, step=4) for i in range(count)]
        activation = st.selectbox('Activación', ['ReLU', 'Tanh'])
        st.caption('La activación permite aprender relaciones no lineales. Las dimensiones de entrada y salida se determinan con el dataset de entrenamiento.')
        architecture = dict(hidden=hidden, activation=activation)
        st.write('Entradas del dataset → ' + ' → '.join(map(str, hidden)) + ' → clases del dataset')
    st.info('La arquitectura seleccionada se utiliza al iniciar el próximo entrenamiento. Los modelos ya guardados conservan su propia arquitectura.')
with training_tab:
    st.header('Entrena y evalúa')
    if kind == 'cnn':
        render_training(architecture)
    else:
        train_dense(architecture)
    runs = saved_runs(kind)
    if runs:
        latest = st.session_state.get('last_cnn' if kind == 'cnn' else 'last_run')
        run = st.selectbox('Resultados de entrenamiento', runs, index=runs.index(latest) if latest in runs else 0,
                           format_func=run_label)
        show_metrics(run)
    else:
        st.info('Aquí aparecerán las métricas cuando entrenes una red.')
with execution_tab:
    st.header('Ejecuta una red entrenada')
    st.write('Selecciona un modelo guardado y proporciona una entrada nueva para obtener su predicción.')
    runs = saved_runs()
    if not runs:
        st.info('Todavía no hay modelos guardados. Configura una arquitectura y entrena una red primero.')
    else:
        run = st.selectbox('Modelo para ejecutar', runs, format_func=run_label)
        config, _, _ = load_run(run)
        with st.expander('Configuración del modelo que se ejecutará'):
            st.json(config)
        if config.get('kind') == 'cnn':
            render_prediction(run, config)
        else:
            predict_dense(run, config)
