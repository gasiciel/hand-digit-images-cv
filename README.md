# Klasyfikacja ręcznie rysowanych i zdigitalizowanych obrazków

Projekt ten zawiera implementację modelu Computer Vision do klasyfikacji 10 klas obrazków, z czego każda klasa składa się z 3 rodzajów obrazków:
* obrazki rysowane ręcznie i fotografowane
* obrazki stworzone przy użyciu stempli i sfotografowane
* obrazy zeskanowane bezpośrednio

Proces obejmuje analizę danych, trening kilku wariantów modeli CNN (w tym z wykorzystaniem transfer learningu), następnie porównanie wyników z trenowaniem z augmentacją danych, prezentację activation maps oraz wdrożenie najlepszego modelu w formie prostej aplikacji webowej Gradio.

Zbiór danych użyty w projekcie to ["Simple hand-drawn and digitized images"](https://www.kaggle.com/datasets/gergvincze/simple-hand-drawn-and-digitized-images/data) z serwisu Kaggle.

## Struktura projektu

-   `data/`: Katalog zawierający zbiór danych (obrazki)
-   `figures/`: Wykresy i wizualizacje wygenerowane w trakcie analizy
-   `model.ipynb`: Jupyter Notebook z całym - od eksploracji danych po ewaluację modeli i wizualizacje Grad-CAM
-   `app.py`: Skrypt uruchamiający aplikację webową Gradio
-   `resnet18.pth`: Zapisane wagi najlepszego, wytrenowanego modelu
-   `requirements.txt`: Plik z listą wszystkich zależności projektu

## Instalacja

Aby uruchomić projekt lokalnie, postępuj zgodnie z poniższymi krokami:

**1. Klonowanie repozytorium**

```bash
git clone https://github.com/gasiciel/hand-drawn-images-cv
cd hand-drawn-images-cv
```

**2. Stworzenie i aktywacja środowiska wirtualnego**


```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Instalacja zależności**

Zainstaluj wszystkie wymagane pakiety za pomocą pip:

```bash
pip install -r requirements.txt
```

## Uruchomienie

Projekt składa się z dwóch głównych części: notebooka oraz aplikacji webowej.

### 1. Analiza i trening w Jupyter Notebook

Aby prześledzić cały proces analizy danych, treningu i ewaluacji modeli, uruchom Jupyter Notebook:

```bash
jupyter notebook
```

W przeglądarce, która się otworzy, wybierz i uruchom plik `model.ipynb`.

### 2. Aplikacja webowa Gradio

Aplikacja webowa pozwala na interaktywną klasyfikację obrazów za pomocą najlepszego wytrenowanego modelu (ResNet18). Można w niej wgrać własny plik lub narysować obrazek odręcznie.

Aby ją uruchomić, wykonaj w terminalu polecenie:

```bash
python app.py
```

Następnie otwórz w przeglądarce podany w terminalu adres.