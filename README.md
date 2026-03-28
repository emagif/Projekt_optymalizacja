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


## Użytkowanie aplikacji :hammer:

Wszelkie algorytmy i funkcje związane z podstawowymi obliczeniami (obliczanie wartości funkcji celu, obliczenie gradientów w punkcie), zostały zaimplementowane w pliku ```algorithms.py```. Funkcje obliczające wartości funkcji celu oraz gradientu funkcji celu zostały nazwane z wykorzystaniem odpowiednich przedrostków i zrozumiałych aliasów, np.: 

```bash
def rosenbrock_f(xk):
    return (1-xk[0])**2 + 100*(xk[1] - xk[0]**2)**2
```

czy,

```bash 
def grad_himmelblau(xk): 
    grad_x1 = 4 * xk[0] * ((xk[0]**2) + xk[1] - 11) + 2 * (xk[0] + (xk[1]**2) - 7)
    grad_x2 = 4 * xk[1] * (xk[0] + (xk[1]**2)-7) + 2 * ((xk[0]**2) + xk[1] - 11)
    return np.array([grad_x1, grad_x2])
```
Następnie zaimplementowane zostały metody Quasi-Newtonowskie (BFGS oraz DFP). Struktura funkcji jest następująca: 

```bash 
def Quasi_Newton_BFGS(start_x1, start_x2, function):
    .
    .
    .
    return best_result, xk_1[0], xk_1[1], i
```
zatem funkcja przyjmuje dwa punkty startowe oraz optymalizowaną wartość funkcji celu. Przy czym jeżeli użytkownik chce wywołać optymalizację dla funkcji Rosenbrocka powinien wpisać nr 1, dla funkcji Three-Hump Camel nr 2, a dla funkcji Himmelblaua nr 3. Funkcja zwraca najlepsze rozwiązanie, punkty dla których jest ona przyjmowana oraz liczbę iteracji potrzebną do osiągnięcia rezultatu końcowego. 


## Autorzy :brain:

Autorami projektu są **Jakub Hinca** oraz **Emanuel Gifuni**. 

*Automatyka, robotyka i systemy sterowania - I semestr studiów magisterskich.*