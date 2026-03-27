# Projekt optymalizacja :rocket:

Repozytorium jest poświęcone projektowi w ramach przedmiotu Metody Optymalizacji. Zaimplementowany został algorytm ... pozwalający na optymalizację jednej z zaproponowanych funkcji celu. 

## Struktura repozytorium :artificial_satellite:

```text
Projekt_optymalizacja/
├── env/
├── main/
│   ├── main.py
│   ├── algorithms.py
│   └── draw_func.py
├── requirements.txt
└── README.md
```



## Instalacja :bulb:

Autorzy zakładają, że pobierający stosują system operacyjny Linux lub korzystają z powłoki Git Bash dla systemów Windows. W przypadku innej konfiguracji prosimy o kontakt.

W pierwszej kolejności należy sklonować repozytorium na swoją maszynę (zalecane użycie SSH) z wykorzystaniem komendy: 

```bash
git clone git@github.com:emagif/Projekt_optymalizacja.git
```

Po skopiowaniu repozytorium należy udać się wewnątrz drzewa projektu: 

```bash 
cd Projekt_optymalizacja
```

i utworzyć wirtualne środowisko: 

```bash
python -m venv env
```
Po utworzeniu środowiska należy je aktywować: 

```bash
source env/Scripts/activate
```

W pliku ```requirements.txt``` znajdują się nazwy wszystkich bibliotek zastosowanych przez autorów projektu, aby zainstalować stosowane przez autorów wersje bibliotek należy użyć komendy: 

```bash 
pip install -r requirements.txt
```
Nie jest sugerowane instalowanie "z ręki" tych samych pakietów, które są zawarte w ```requirements.txt```. Najlepiej zastosować się do powyższych zaleceń. 


## Autorzy :brain:

Autorami projektu są **Jakub Hinca** oraz **Emanuel Gifuni**. 

*Automatyka, robotyka i systemy sterowania - I semestr studiów magisterskich.*