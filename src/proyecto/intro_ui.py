"""Presentación didáctica previa al flujo práctico del laboratorio."""

from pathlib import Path

import streamlit as st


_IMAGE_DIR = Path(__file__).parent / 'assets' / 'intro'

_SLIDES = [
    {
        'cover': True,
    },
    {
        'title': '¿Qué es una red neuronal?',
        'image': 'neural-network-overview.png',
        'caption': 'Esquema general de capas y conexiones en una red neuronal artificial.',
        'paragraphs': [
            'Es un modelo matemático que aprende relaciones entre datos de entrada y '
            'respuestas. Está formado por unidades conectadas que se organizan en capas.',
            'Durante el entrenamiento, el modelo ajusta sus parámetros usando ejemplos. '
            'Después puede aplicar lo aprendido a datos nuevos similares.',
        ],
        'flow': [
            ('Datos de entrada', 'Medidas, características o píxeles.'),
            ('Capas conectadas', 'Transforman las señales y extraen patrones.'),
            ('Respuesta', 'Predicción para una o varias clases.'),
        ],
        'note': 'La inspiración biológica es una analogía: una red artificial es un '
                'programa matemático y no una copia del cerebro.',
    },
    {
        'title': '¿Qué es una neurona?',
        'image': 'biological-neuron.png',
        'caption': (
            'Ilustración de una neurona biológica; se muestra como inspiración, '
            'no como equivalencia.'
        ),
        'paragraphs': [
            'Una neurona biológica es una célula del sistema nervioso. Sus dendritas '
            'reciben señales, el cuerpo celular las integra y el axón transmite una señal.',
            'Una neurona artificial es una unidad de cálculo mucho más sencilla. Recibe '
            'números y produce otro número para que otras unidades puedan procesarlo.',
        ],
        'bullets': [
            '**1 · Dendritas:** reciben señales.',
            '**2 · Cuerpo celular:** integra señales.',
            '**3 · Núcleo:** contiene información de la célula.',
            '**4 · Axón:** transmite señales a otras células.',
        ],
        'note': 'La imagen representa una célula real. En la siguiente página veremos '
                'el modelo matemático artificial usado por la red.',
    },
    {
        'title': '¿Qué es una neurona artificial?',
        'image': 'artificial-neuron-detail.png',
        'caption': (
            'Esquema matemático de una neurona artificial: entradas, combinación '
            'y salida.'
        ),
        'paragraphs': [
            'Es una unidad de cálculo que recibe valores numéricos, los combina con '
            'parámetros aprendidos y produce una señal de salida. Esa señal puede '
            'convertirse en entrada para otras neuronas.',
            'A diferencia de una neurona biológica, no es una célula: es una operación '
            'matemática sencilla. La red completa obtiene su capacidad al conectar '
            'muchas unidades y ajustar sus parámetros con ejemplos.',
        ],
        'bullets': [
            '**Entradas (x):** características del ejemplo, como medidas o píxeles.',
            '**Pesos (w):** regulan cuánto influye cada entrada.',
            '**Sesgo (b):** permite ajustar el umbral de respuesta.',
            '**Activación (f):** transforma la suma en la señal de salida.',
        ],
        'note': 'Los pesos y el sesgo se modifican al entrenar; durante la ejecución '
                'se mantienen fijos para producir predicciones.',
    },
    {
        'title': '¿Cómo funciona una neurona artificial?',
        'image': 'neuron.png',
        'caption': 'Entradas, pesos, sesgo, suma y función de activación.',
        'paragraphs': [
            'Cada entrada describe una característica del ejemplo. La neurona '
            'multiplica cada una por un peso, suma los resultados y añade un sesgo.',
            'La función de activación transforma esa suma en una señal para la siguiente '
            'capa. Así, una operación sencilla se convierte en parte de un modelo mayor.',
        ],
        'formula': r'z = w_1x_1 + w_2x_2 + \cdots + w_mx_m + b, '
                   r'\qquad a = f(z)',
        'note': 'Los pesos determinan cuánto influye cada entrada; el sesgo ajusta el '
                'umbral. Ambos se aprenden durante el entrenamiento.',
    },
    {
        'title': '¿Cómo funciona una red neuronal?',
        'image': 'network-flow.png',
        'caption': 'Capas conectadas: las señales avanzan y el error retrocede al entrenar.',
        'paragraphs': [
            'La capa de entrada recibe los datos, las capas ocultas transforman sus '
            'señales y la capa de salida calcula una respuesta para cada clase.',
            'En la propagación hacia delante, cada capa calcula sus activaciones y las '
            'envía a la siguiente. Al entrenar, la retropropagación calcula cómo cambiar '
            'los parámetros para reducir el error.',
            'Una red densa conecta neuronas entre capas. Una CNN procesa imágenes con '
            'filtros que detectan patrones locales y capas de agrupamiento que reducen '
            'los mapas de características.',
        ],
    },
    {
        'title': 'Las funciones de activación permiten aprender relaciones complejas',
        'image': 'activations.png',
        'caption': 'Respuestas y derivadas de sigmoide, tanh, ReLU y Leaky ReLU.',
        'paragraphs': [
            'Después de sumar las entradas ponderadas, la activación transforma la señal '
            'que pasa a la siguiente capa. Sin activaciones no lineales, muchas capas '
            'seguidas se comportarían como una única transformación lineal.',
        ],
        'bullets': [
            '**ReLU:** deja pasar valores positivos y convierte los negativos en cero.',
            '**Sigmoide:** comprime la salida entre 0 y 1.',
            '**Tanh:** comprime la salida entre −1 y 1.',
        ],
    },
    {
        'title': 'Entrenar reduce poco a poco el error',
        'image': 'gradient-descent.png',
        'caption': 'Trayectoria ilustrativa de descenso por una superficie de pérdida.',
        'paragraphs': [
            'La función de pérdida compara la predicción con la respuesta correcta. '
            'La retropropagación calcula gradientes que indican cómo influyen los '
            'pesos y sesgos en esa pérdida.',
            'Un optimizador usa esos gradientes para actualizar los parámetros. El ciclo '
            'se repite con los ejemplos de entrenamiento; una época es un recorrido '
            'completo por esos ejemplos.',
        ],
        'formula': r'W \leftarrow W - \eta\,\nabla_W\mathcal{L}',
        'note': 'La gráfica ilustra pocos parámetros; una red real tiene una superficie '
                'de pérdida de muchas dimensiones.',
    },
    {
        'title': 'Validar ayuda a saber si la red generaliza',
        'image': 'training-curves.png',
        'caption': 'Ejemplo de evolución de pérdida y exactitud durante el aprendizaje.',
        'paragraphs': [
            'Los datos se separan para cumplir funciones distintas: entrenamiento '
            'actualiza los pesos, validación ayuda a comparar configuraciones y prueba '
            'estima el resultado final con ejemplos reservados.',
            'Al comparar las curvas por época observamos si el modelo sigue mejorando '
            'con ejemplos que no ajustan sus pesos.',
        ],
        'note': 'Si la pérdida de entrenamiento sigue bajando mientras la de validación '
                'sube, la red puede estar sobreajustándose a los ejemplos vistos.',
    },
    {
        'title': 'Ejecutar usa lo aprendido para predecir',
        'paragraphs': [
            'Al ejecutar, se carga un modelo guardado y se le proporciona un ejemplo. '
            'La red aplica el preprocesamiento esperado y recorre sus capas para '
            'producir una predicción, sin volver a ajustar los pesos.',
            'En clasificación, las puntuaciones ayudan a comparar las clases. No '
            'garantizan que una entrada ajena al dataset sea fiable. La visualización '
            'de activaciones permite observar los valores calculados en cada capa para '
            'esa entrada.',
        ],
        'flow': [
            ('Arquitectura', 'Elige MLP o CNN y define sus capas.'),
            ('Entrenamiento', 'Selecciona datos, entrena y revisa las métricas.'),
            ('Ejecución', 'Abre un modelo y pruébalo con ejemplos.'),
        ],
        'note': 'La traza de activaciones ayuda a seguir el recorrido; no mide importancia '
                'causal ni sustituye la evaluación con datos reservados.',
    },
]


def render_introduction():
    """Presenta un concepto por diapositiva y permite recorrer la guía."""
    total = len(_SLIDES)
    current = int(st.session_state.get('intro_slide', 0))
    current = min(max(current, 0), total - 1)
    slide = _SLIDES[current]

    if slide.get('cover'):
        st.markdown(
            '<style>[data-testid="stSidebar"], [data-testid="collapsedControl"], '
            '[data-testid="stHeader"], footer {display:none;} '
            '.block-container {padding-top:0;}</style>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:42vh'></div>", unsafe_allow_html=True)
        _, center_column, _ = st.columns([1, 2, 1])
        with center_column:
            st.markdown(
                "<h1 style='text-align:center'>Redes neuronales desde cero</h1>",
                unsafe_allow_html=True,
            )
            if st.button(
                'Comenzar',
                type='primary',
                use_container_width=True,
                key='intro-start',
            ):
                st.session_state['intro_slide'] = 1
                st.rerun()
        return

    st.title('Redes neuronales desde cero')
    st.caption('Presentación guiada · conceptos para empezar')
    st.progress((current + 1) / total)
    st.caption(f'Diapositiva {current + 1} de {total}')

    st.header(slide['title'])
    if slide.get('image'):
        text_column, image_column = st.columns([0.85, 1.15], gap='large')
    else:
        text_column = st.container()
        image_column = None

    with text_column:
        for paragraph in slide['paragraphs']:
            st.write(paragraph)
        if slide.get('formula'):
            st.latex(slide['formula'])
        for bullet in slide.get('bullets', []):
            st.markdown(f'- {bullet}')
        if slide.get('flow'):
            columns = st.columns(3)
            for column, (title, description) in zip(columns, slide['flow']):
                with column:
                    with st.container(border=True):
                        st.markdown(f'**{title}**')
                        st.write(description)

    if slide.get('image'):
        with image_column:
            st.image(
                str(_IMAGE_DIR / slide['image']),
                caption=slide['caption'],
                use_column_width=True,
            )

    if slide.get('note'):
        st.info(slide['note'])

    previous_column, page_column, next_column = st.columns([1, 1, 1])
    with previous_column:
        previous = st.button(
            '← Anterior',
            disabled=current == 0,
            use_container_width=True,
            key=f'intro-previous-{current}',
        )
    with page_column:
        st.caption(f'{current + 1} / {total}')
    with next_column:
        label = 'Ir al laboratorio →' if current == total - 1 else 'Siguiente →'
        following = st.button(
            label,
            type='primary',
            use_container_width=True,
            key=f'intro-next-{current}',
        )

    if previous:
        st.session_state['intro_slide'] = current - 1
        st.rerun()
    if following:
        if current == total - 1:
            st.session_state['intro_completed'] = True
            st.session_state.pop('intro_slide', None)
        else:
            st.session_state['intro_slide'] = current + 1
        st.rerun()
