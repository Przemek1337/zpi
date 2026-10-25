# Wnioski z eksperymentów

Opis tego, co zrobiliśmy, jakie wyniki dostaliśmy i dlaczego nie da się ich istotnie poprawić w ramach
specyfikacji. Wykresy i tabele do wszystkich liczb są w `tuning_and_test_analysis.ipynb` oraz w
`artifacts/tuning/final_model/test_evaluation/report.md`.

## Wynik w jednym zdaniu

Model wykrywa na zbiorze testowym 84,0% fraudów (199 z 237) przy 0,98% fałszywych alarmów. Cel ze
specyfikacji to co najmniej 85% przy FPR poniżej 1%, więc brakuje 3 wykrytych fraudów. Formalnie kryterium
nie jest spełnione, ale różnica mieści się w niepewności pomiaru.

## Co zrobiliśmy

1. **Dane.** 284 807 transakcji, w tym 492 fraudy. Usunęliśmy 1 081 duplikatów, zostało 283 726.
   `Time` zamieniliśmy na porę doby (sin i cos godziny), bo surowa liczba sekund od początku zbioru nic nie
   znaczy poza tym zbiorem. `Amount` przeszło przez log(1 + x), żeby ograniczyć wpływ bardzo dużych kwot.
   Daje to 31 cech.
2. **Podział.** Transakcje prawidłowe podzieliliśmy 70/15/15 na train, walidację i test, fraudy po połowie na
   walidację i test. Train zawiera wyłącznie transakcje prawidłowe (198 277), walidacja 42 724 (236 fraudów),
   test 42 725 (237 fraudów). Skalowanie (StandardScaler) dopasowane tylko na train. Seed 42.
3. **Model.** Głęboki autoenkoder w PyTorch, funkcja celu MSE, trening wyłącznie na transakcjach
   prawidłowych. Wynikiem dla transakcji jest jej błąd rekonstrukcji (MSE).
4. **Próg.** Najniższy próg MSE, przy którym na walidacji mniej niż 1% transakcji prawidłowych jest
   oznaczanych jako fraud. Próg wyznaczamy na walidacji i bez zmian stosujemy na teście.
5. **Pierwszy model** z ręcznie dobranymi parametrami (warstwy 24 i 16, latent 8, leaky_relu, 2 807
   parametrów): 77,1% recall na walidacji, 81,9% na teście.
6. **Strojenie (Optuna).** 9 godzin na karcie NVIDIA L4, trzy próby równolegle. Optuna maksymalizowała recall
   na walidacji przy FPR < 1%. Przeszukiwała liczbę warstw (1-4), ich szerokość (16-128 neuronów), rozmiar
   bottlenecku (2-16), aktywację (6 wariantów), dropout, batch norm, wariant denoising, optymalizator, learning
   rate, weight decay, batch size, gradient clipping i scheduler. Razem 717 prób: 411 zakończonych, 306
   przerwanych wcześnie jako wyraźnie słabe, 0 błędów.
7. **Model finalny.** Najlepsza konfiguracja wytrenowana ponownie przez 100 epok: warstwy 96, 76 i 60,
   latent 13, aktywacja elu, batch norm, dropout 0,13, AdamW z learning rate 3,8e-3, batch 1024, bez
   denoisingu, 32 686 parametrów. Recall na walidacji 80,5%.
8. **Test.** Jednorazowa ocena modelu finalnego na zbiorze testowym i porównanie z Isolation Forest
   wytrenowanym w tych samych warunkach (tylko transakcje prawidłowe, próg wyznaczony tą samą metodą).

## Wyniki na zbiorze testowym

| | Pierwszy model | Model finalny | Isolation Forest |
|---|---|---|---|
| Recall | 81,9% (194/237) | **84,0% (199/237)** | 53,2% (126/237) |
| FPR | 0,99% | **0,98%** | 1,00% |
| Precision | 31,6% | 32,3% | 23,0% |
| F1 | 0,456 | 0,467 | 0,321 |
| PR-AUC | 0,568 | **0,745** | 0,271 |

Macierz pomyłek modelu finalnego: 199 wykrytych fraudów, 38 niewykrytych, 417 fałszywych alarmów,
42 071 poprawnie przepuszczonych transakcji prawidłowych.

Precision 32% oznacza, że na każde trzy alarmy jeden jest prawdziwym fraudem. Przy celu FPR < 1% i fraudach
stanowiących 0,17% transakcji wyższej precyzji nie da się uzyskać: nawet przy 100% recall fałszywe alarmy
byłyby liczniejsze niż fraudy.

**Niepewność wyniku.** Recall liczony na 237 fraudach ma 95% przedział ufności od 78,8% do 88,1%. Cel 85%
leży wewnątrz, więc wyniku nie da się statystycznie odróżnić od celu. Mimo to raportujemy go jako
niespełniony, bo tak wynika z liczb.

## Gdzie jest sufit

Na walidacji najlepsze konfiguracje zatrzymują się na 80-82% recall, a na teście na około 84%.
Wskazują na to trzy niezależne obserwacje:

- **Strojenie się wypłaszcza.** Krzywa najlepszego wyniku w historii optymalizacji szybko przestaje rosnąć.
  Setki późniejszych prób poprawiały wynik o pojedyncze fraudy albo wcale.
- **Bardzo różne architektury dochodzą do tego samego poziomu.** Najlepszy recall na walidacji dla jednej
  warstwy to 80,1%, dla dwóch 80,9%, dla trzech 82,2%, dla czterech 81,4%. Cała rozpiętość to 2 punkty
  procentowe, czyli około 5 fraudów.
- **Isolation Forest nie wykrywa żadnego z fraudów, które przegapił autoenkoder.** Wszystkie 38 niewykrytych
  przez model finalny jest niewykrytych także przez Isolation Forest, który działa na zupełnie innej zasadzie
  (izoluje rzadkie punkty, nie odtwarza danych). Gdyby problem leżał w modelu, druga metoda łapałaby część tych
  przypadków.

## Dlaczego sufit jest w tym miejscu

Część fraudów w tym zbiorze wygląda dokładnie jak zwykłe transakcje. Dobrze to widać w szczegółowej analizie
pierwszego modelu (dla modelu finalnego obraz jest taki sam, liczby są w jego `report.md`):

- Wykryte fraudy mają medianę MSE 10,1, transakcje prawidłowe 0,41, a niewykryte fraudy 0,71. Dziesięć
  najtrudniejszych przypadków ma MSE 0,23-0,43, czyli tyle co typowa transakcja prawidłowa albo mniej.
- Wykryte fraudy odróżnia ogromny błąd rekonstrukcji na kilku cechach: V17 (średnio 190 wobec 0,5 dla
  transakcji prawidłowych), V14 (94), V12 (75), V16 (52), V10 (49). Niewykryte fraudy mają na tych cechach
  błąd 1-3, prawie taki jak transakcje prawidłowe. Brakuje im sygnału, po którym model rozpoznaje fraudy.
- Obniżenie progu nie rozwiązuje problemu. W pierwszym modelu osiągnięcie 85% recall wymagałoby progu
  dającego 5,3% fałszywych alarmów zamiast 1%, czyli ponad pięć razy więcej. Na krzywej precision-recall widać
  to jako gwałtowny spadek precyzji tuż przed recall 85%.

Każda metoda uczona wyłącznie na transakcjach prawidłowych, która ocenia, jak bardzo transakcja od nich
odbiega, uzna takie fraudy za normalne. Ograniczenie leży w danych: 31 cech z dwóch dni, w większości po
anonimizującej transformacji PCA, bez identyfikatora klienta i bez jego historii. Fraud, który w tych cechach
nie odstaje od typowej transakcji, jest dla tego podejścia niewidoczny.

## Co dało strojenie, a co nie

- **Zysk.** Recall na teście wzrósł z 81,9% do 84,0% (5 fraudów więcej przy tym samym FPR), PR-AUC z 0,568 do
  0,745. Ten drugi wzrost jest większy, bo model finalny lepiej porządkuje transakcje według podejrzaności w
  całym zakresie, a nie tylko przy wybranym progu.
- **Większa sieć pomogła.** Zwycięski model ma 32 686 parametrów wobec 2 807 w pierwszym. Najlepiej
  wypadły trzy warstwy ukryte (82,2%). Czwarta warstwa nic nie dodała (81,4%), a jedna lub dwie dawały
  najwyżej 80,1-80,9%.
- **Największy wpływ** na wynik miały learning rate (16% zmienności wyników), dropout (14%), liczba warstw
  (12%) i tempo zwężania warstw (12%), czyli parametry treningu i kształt sieci.
- **Aktywacja.** elu i gelu dawały podobne wyniki (średnio 80,3% na walidacji). leaky_relu z pierwszego modelu
  wypadała najsłabiej (średnio 72,9%) i Optuna szybko ją porzuciła.
- **Bez znaczenia** okazały się denoising (łącznie około 1,5% zmienności, najlepszy wynik z szumem 81,8%, bez
  szumu 82,2%), batch norm i batch size.
- **Wynik najlepszej próby (82,2%) jest zawyżony.** To maksimum z 411 prób, a na walidacji jeden fraud to 0,42
  punktu procentowego, więc część przewagi to szczęśliwy losowy układ. Ta sama konfiguracja wytrenowana
  ponownie dała 80,5%. Dlatego bardziej wiarygodne jest 80,5%.

## Pozostałe wnioski

- **Hipoteza się potwierdza.** Autoenkoder uczony na transakcjach prawidłowych daje fraudom średnio wielokrotnie
  wyższy błąd rekonstrukcji (średnie MSE fraudów na walidacji około 20, transakcji prawidłowych około 0,15-0,6
  zależnie od modelu). Na tej podstawie wykrywa większość fraudów.
- **Autoenkoder jest wyraźnie lepszy od baseline.** Przy tym samym FPR wykrywa o 30,8 punktu procentowego
  więcej fraudów niż Isolation Forest i ma prawie trzy razy wyższe PR-AUC.
- **ROC-AUC wprowadza w błąd przy takich danych.** Dla pierwszego modelu i Isolation Forest wynosiło
  odpowiednio 0,947 i 0,951, prawie tyle samo, choć w punkcie pracy (FPR < 1%) recall różni się o prawie 30
  punktów. ROC-AUC uśrednia po wszystkich progach, także tych z FPR rzędu kilkudziesięciu procent, które w
  praktyce nas nie interesują. Do porównań lepiej nadają się recall przy ustalonym FPR i PR-AUC.
- **Walidacja i test różnią się o kilka punktów.** Oba modele miały na teście wyższy recall niż na walidacji
  (81,9% wobec 77,1% i 84,0% wobec 80,5%). To nie błąd, tylko efekt małej liczby fraudów w każdym zbiorze
  (około 236). Pojedyncze fraudy przesuwają wynik o 0,4 punktu, więc różnice rzędu 1-2 punktów między
  konfiguracjami trzeba traktować ostrożnie.
- **Metodologia.** Zbiór testowy został użyty raz. Próg i hiperparametry wybieraliśmy wyłącznie na walidacji.
  Dlatego nie poprawiamy już modelu ani progu pod wynik testu: wtedy test przestałby być niezależny, a o to
  chodzi w kryterium ze specyfikacji.
- **VAE nie testowaliśmy.** Jego funkcja celu to MSE plus dywergencja KL, a specyfikacja wymaga czystego MSE.
  Nie spodziewamy się też, że przesunąłby sufit, bo uczy się tego samego obrazu „normalnej” transakcji z tych
  samych cech.

## Wydajność

- Trening modelu finalnego: 178 s (100 epok) na NVIDIA L4.
- Ocena 42 725 transakcji testowych: 0,2 s na GPU.
- Isolation Forest: trening 2 s.
- Strojenie: 9 godzin na trzech równoległych procesach.

## Ograniczenia i możliwe kierunki

- Zbiór obejmuje dwa dni i nie ma identyfikatora klienta. Nie da się więc wykorzystać tego, co w praktyce
  najlepiej zdradza fraud: odstępstwa od zwykłego zachowania konkretnego klienta (typowe kwoty, pory, miejsca).
- Cechy V1-V28 po PCA nie mają znaczenia biznesowego, więc nie da się wyjaśnić, dlaczego dana transakcja
  dostała alarm.
- Uczenie z wykorzystaniem etykiet fraudów prawdopodobnie poprawiłoby wynik, ale specyfikacja wymaga treningu
  wyłącznie na transakcjach prawidłowych.
- Ocena opiera się na około 240 fraudach w każdym zbiorze, więc wszystkie wyniki obarczone są niepewnością rzędu
  kilku punktów procentowych.

## Gdzie co leży

| Co | Gdzie |
|---|---|
| Analiza z wykresami | `notebooks/tuning_and_test_analysis.ipynb` |
| Raport z testu (metryki, macierze, FN, czasy) | `artifacts/tuning/final_model/test_evaluation/report.md` |
| Wszystkie próby strojenia | `artifacts/tuning/trials.csv` |
| Konfiguracja zwycięzcy | `artifacts/tuning/best_config.yaml` |
| Model finalny | `artifacts/tuning/final_model/model.pt` |
