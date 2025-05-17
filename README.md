
Hello

### Hausaufgabe bis 21.05

Erstelle ein Python-Programm, das eine Grafik erzeugt, in der die semantischen Nachbarn eines gegebenen Wortes sowie deren Nachbarn visualisiert werden. Dabei soll der Input ein beliebig wählbares Wort sein, der Out die Grafik des resultierenden Netzes.

Mein Vorschlag für die Pipeline:

- Input
  - Wort
  - Anzahl der Nachbarn
  - Anzahl der Schichten
- Nachbarn mit Modell suchen -> d.h. die Hausi vom letzten mal erweitert auf die Anzahl der Schichten
- Visualisierung
  - Welche lib ist dafür am besten geeignet?
 
Fragen:
- Bei meinen Beispiel Wörtern sah man ja letztes mal, dass die Flexionen zunächst die höchsten Werte einnehmen. Wäre es sinnvoll diese auszuklammern, dass wir quasi die "wahren" Nachbarn bekommen?
- Welche lib für die Visualisierung und wir müssen dafür die Daten aufbereitet werden?
  - Ich habe networkx gefunden, damit kann man Netzwerke aufbauen/visualisieren. Hat aber anscheinend Probleme mit größeren Netzwerken, wir müssen also bisschen schaun, wies klappt.
