# Jarvis - vision par ordinateur et HUD moto

Application Python qui traite une video ou une webcam et affiche :

- les voitures en vert, les motos en rouge et les autres classes de vehicules dans des couleurs distinctes ;
- une zone centrale de circulation (ROI), un compteur et une alerte de prudence si le nombre de vehicules atteint le seuil ;
- la detection et la lecture locale des plaques avec FastALPR ;
- une fleche rouge qui signale la section la plus dense de la zone ROI.

L'analyse de la ROI est une estimation visuelle, pas une mesure de distance, de vitesse ou de securite. L'alerte recommande de ralentir et de garder ses distances ; elle ne conseille pas de manoeuvre ou de changement de voie. Ne pas utiliser l'application comme substitut a l'attention du conducteur.

## Installation

Python 3.10 ou plus recent est recommande. Depuis le dossier du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --force-reinstall --no-deps opencv-python
```

La derniere commande restaure le paquet OpenCV avec interface graphique, necessaire pour l'affichage en direct sous Windows (FastALPR installe aussi la variante sans interface).

Les modeles de detection des vehicules et des plaques sont telecharges automatiquement au premier lancement s'ils ne sont pas deja presents. Une connexion Internet est necessaire a cette etape.

La detection et la lecture des plaques s'effectuent localement ; l'application n'envoie pas les images a un service distant. La reconnaissance depend de la visibilite, de la taille et de l'angle de la plaque.

La fleche rouge pointe vers la section la plus dense de la ROI pour signaler le trafic. Elle ne conseille ni changement de voie, ni manoeuvre.

## Utilisation

Webcam par defaut :

```powershell
python -m jarvis_vision.app
```

Video locale, seuil de congestion et video annotee :

```powershell
python -m jarvis_vision.app --source "C:\videos\trajet.mp4" --threshold 4 --output "C:\videos\jarvis-sortie.mp4"
```

Sans lecture des plaques :

```powershell
python -m jarvis_vision.app --source 0 --no-plates
```

Appuyer sur `q` ou `Echap` pour quitter. `--no-display` traite la source sans fenetre et peut etre combine avec `--output`. Les plaques sont analysees toutes les huit images par defaut (`--ocr-interval`).

## Tests

```powershell
python -m unittest discover -s tests -v
```
