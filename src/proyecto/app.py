"""Interfaz didáctica escrita en Python."""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

from proyecto.training import load_run, predict, prepare, train

st.set_page_config(page_title="Laboratorio de redes neuronales", page_icon="🧠", layout="wide")
st.title("Laboratorio de redes neuronales")
st.caption("Configura · comprende · entrena · experimenta")
st.info("Redes densas con PyTorch y convolucionales con TensorFlow. Ejecución en CPU dentro de Docker.")

with st.sidebar:
    st.header("Tu laboratorio")
    st.write("Una red aprende de ejemplos etiquetados. Después la probamos con datos que no vio.")
    st.caption("Los modelos y los datos se conservan en el volumen de Docker.")

setup, results, cnn_tab = st.tabs(["1 · Red densa", "2 · Resultados de redes densas", "3 · CNN y datasets públicos"])
with setup:
    st.subheader("Datos de aprendizaje")
    source = st.radio("Origen", ["Iris: ejemplo incluido", "Mi archivo CSV"], horizontal=True)
    frame = None
    if source.startswith("Iris"):
        iris = load_iris(as_frame=True)
        frame = iris.data.copy()
        frame["especie"] = [iris.target_names[i] for i in iris.target]
        st.write("150 flores, cuatro medidas y tres especies. La red aprenderá a clasificar flores.")
    else:
        st.caption("CSV con encabezados, entradas numéricas y una columna de clase. Máximo 20 MB.")
        upload = st.file_uploader("Selecciona un CSV", type=["csv"])
        if upload:
            try:
                frame = pd.read_csv(upload)
            except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
                st.error(f"No se pudo leer el CSV: {exc}")
    if frame is not None and len(frame.columns):
        st.dataframe(frame.head(10), use_container_width=True)
        target = st.selectbox("Columna que la red debe predecir", frame.columns, index=len(frame.columns)-1)
        try:
            _, _, columns, classes = prepare(frame, target)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success(f"Datos compatibles: {len(frame)} ejemplos · {len(columns)} entradas · {len(classes)} clases")
            with st.form("architecture"):
                st.subheader("Arquitectura de la red densa")
                count = st.slider("Capas ocultas", 1, 4, 2)
                st.caption("Se usan las primeras N capas de los controles siguientes.")
                widths = [st.slider(f"Neuronas en capa {i+1}", 4, 128, 32, step=4) for i in range(4)]
                activation = st.selectbox("Activación", ["ReLU", "Tanh"])
                st.caption("La activación permite aprender relaciones no lineales entre las entradas y las clases.")
                epochs = st.slider("Épocas: recorridos por los datos de entrenamiento", 5, 300, 50)
                batch = st.selectbox("Ejemplos por lote", [8, 16, 32, 64, 128], index=2)
                rate = st.selectbox("Tasa de aprendizaje", [0.0001, 0.001, 0.01], index=1)
                seed = st.number_input("Semilla para repetir el experimento", 0, 2147483647, 42)
                st.write("Optimizador: Adam · Pérdida: entropía cruzada · Salida: una probabilidad por clase.")
                st.write("Separación: 60 % entrenamiento, 20 % validación, 20 % prueba. La normalización aprende solo del entrenamiento.")
                submitted = st.form_submit_button("Entrenar y guardar", type="primary")
            if submitted:
                st.write(f"Arquitectura: {len(columns)} → {' → '.join(map(str, widths[:count]))} → {len(classes)}")
                bar, chart = st.progress(0), st.empty()
                def update(history):
                    bar.progress(len(history) / epochs)
                    chart.line_chart(pd.DataFrame(history).set_index("época"))
                try:
                    with st.spinner("La red está aprendiendo…"):
                        run_id = train(frame, target, widths[:count], activation, epochs, rate, batch, int(seed), update)
                    st.session_state["last_run"] = run_id
                    st.success("Modelo guardado. Abre Resultados y predicciones para probarlo.")
                except (ValueError, OSError, RuntimeError) as exc:
                    st.error(f"No se completó el entrenamiento: {exc}")

with results:
    root = Path(os.environ.get("NN2GUI_DATA", "data"))
    runs = sorted([p.name for p in root.glob("*") if (p / "history.json").exists() and load_run(p.name)[0].get("kind") != "cnn"], reverse=True)
    if not runs:
        st.info("Entrena tu primera red para ver los resultados aquí.")
    else:
        latest = st.session_state.get("last_run")
        run_id = st.selectbox("Experimento guardado", runs, index=runs.index(latest) if latest in runs else 0)
        config, metrics, history = load_run(run_id)
        st.metric("Exactitud en prueba: ejemplos nuevos correctamente clasificados", f"{metrics['accuracy']:.1%}")
        st.caption("El conjunto de prueba no se usa para ajustar pesos. Evita elegir configuraciones por esta cifra; usa la validación.")
        st.line_chart(pd.DataFrame(history).set_index("época"))
        st.write("Si la pérdida de entrenamiento baja y la de validación sube, puede haber sobreajuste.")
        st.write("Matriz de confusión: filas = clase real; columnas = clase predicha.")
        st.dataframe(pd.DataFrame(metrics["confusion"], index=config["classes"], columns=config["classes"]))
        with st.expander("Configuración y particiones del experimento"):
            st.json(config)
        st.subheader("Prueba tu red")
        st.caption("Introduce las mismas medidas y unidades del dataset. La red solo conoce sus clases de entrenamiento.")
        with st.form("prediction"):
            values = [st.number_input(name, value=float(mean), key=f"{run_id}-{i}")
                      for i, (name, mean) in enumerate(zip(config["columns"], config["mean"]))]
            apply = st.form_submit_button("Predecir")
        if apply:
            try:
                probabilities = predict(run_id, values)
                st.success(f"Clase predicha: {config['classes'][int(probabilities.argmax())]}")
                st.bar_chart(pd.DataFrame({"Probabilidad": probabilities}, index=config["classes"]))
                st.caption("Estas probabilidades no garantizan fiabilidad para entradas ajenas al dataset.")
            except (ValueError, OSError, RuntimeError) as exc:
                st.error(str(exc))

with cnn_tab:
    from proyecto.cnn_ui import render
    render()
