from . import __version__, list_gpus, select_device


def main() -> None:
    print(f"proyecto {__version__}")
    gpus = list_gpus()
    print("GPU detectadas:", ", ".join(map(str, gpus)) if gpus else "ninguna")
    print("Dispositivo seleccionado:", select_device())


if __name__ == "__main__":
    main()
