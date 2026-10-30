#!/usr/bin/env bash
# Crea ~/Escritorio/proyecto (o ~/Desktop), inicializa git y, si hay `gh`, lo sube a GitHub.
set -euo pipefail
NOMBRE="${1:-proyecto}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DESK="$HOME/Escritorio"; [ -d "$DESK" ] || DESK="$HOME/Desktop"
DEST="$DESK/$NOMBRE"
mkdir -p "$DEST"
cp -a "$SRC"/. "$DEST"/
rm -rf "$DEST/.git"
cd "$DEST"
git init -b main
git add -A
git commit -m "Base inicial del proyecto"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh repo create "$NOMBRE" --private --source=. --remote=origin --push
else
  echo "Instala/autentica gh (gh auth login) y ejecuta: gh repo create $NOMBRE --private --source=. --push"
fi
echo "Listo: $DEST"
