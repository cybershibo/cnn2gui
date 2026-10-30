"""Entradas de consola para entrenar y aplicar el clasificador."""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Caso práctico CNN para clasificar prendas")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("device", help="Comprueba qué GPU detecta TensorFlow")
    download = commands.add_parser("download", help="Descarga y comprueba Fashion-MNIST")
    download.add_argument("--data", default="data/fashion-mnist", help="Carpeta local del dataset")
    train = commands.add_parser("train", help="Entrena, evalúa y genera el informe")
    train.add_argument("--out", required=True, help="Carpeta nueva para los resultados")
    train.add_argument("--data", default="data/fashion-mnist", help="Carpeta descargada del dataset")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--sample-index", type=int, default=0,
                       help="Imagen de prueba que se exporta para la demostración")
    predict = commands.add_parser("predict", help="Clasifica un archivo PNG o JPG")
    predict.add_argument("--run", required=True, help="Carpeta de un entrenamiento previo")
    predict.add_argument("--image", required=True, help="Imagen a clasificar")
    export = commands.add_parser("export", help="Exporta PNG variados del test oficial y sus etiquetas")
    export.add_argument("--data", default="data/fashion-mnist", help="Carpeta local del dataset")
    export.add_argument("--out", default="outputs/moda-01/imagenes-prueba", help="Carpeta nueva para los PNG")
    export.add_argument("--per-class", type=int, default=5, help="Imágenes por cada una de las 10 clases")
    export.add_argument("--seed", type=int, default=42, help="Semilla para seleccionar ejemplos")
    export.add_argument("--skip-index", type=int, default=0, help="Índice de test a excluir (entrada.png por defecto)")
    args = parser.parse_args(argv)

    if args.command == "device":
        import tensorflow as tf
        print(json.dumps({"tensorflow": tf.__version__,
                          "gpus": [gpu.name for gpu in tf.config.list_physical_devices("GPU")]},
                         indent=2))
    elif args.command == "download":
        from .dataset import download_dataset
        download_dataset(args.data)
    elif args.command == "train":
        from .pipeline import train_case
        train_case(args.out, args.epochs, args.seed, args.sample_index, args.data)
    elif args.command == "export":
        from .export import export_examples
        export_examples(args.data, args.out, args.per_class, args.seed, args.skip_index)
    elif args.command == "predict":
        from .pipeline import predict_file
        print(json.dumps(predict_file(args.run, args.image), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
