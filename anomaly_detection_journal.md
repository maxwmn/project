# Anomaly Detection — Verlaufsjournal

Chronologisches Protokoll: was in den `anomaly_detection`-Skripten gemacht wurde und warum.

## Phase 1 — Baseline-Modell

- Ziel: ML-Programm, das Turbinen-Anomalien aus SCADA-Sensordaten (Windgeschwindigkeit, Temperaturen, Leistung, etc.) vorhersagt.
- Bestehenden Code geprüft, bevor etwas Neues gebaut wurde: `risk_analysis.py` existierte bereits, `src/` war jedoch Altlast aus einer anderen Aufgabe (Wasseraufbereitung) und irrelevant.
- Entscheidung: frischer, eigenständiger Ansatz statt Wiederverwendung von `src/`.
- `anomaly_detection.py` gebaut: Random Forest, gepoolt über alle 22 Turbinen, Label = `status_type_id` ungleich 0/2 (laut Datensatz-Doku "normal"/"idling").
- Speicherabsturz beim vollen Datensatz (1,2 Mio. Zeilen). Ursache gefunden: `max_samples` war definiert, aber nie tatsächlich ans Modell übergeben — der Random Forest bootstrappte versehentlich den vollen Datensatz pro Baum. Bug behoben, zusätzlich `float32`-Downcasting und ein Dev-Modus (`SAMPLE_EVERY_NTH_ROW`) für schnelle Testläufe ergänzt.
- Ergebnis nach Fix: Accuracy 0,71, F1 0,57 (gepoolt).

## Phase 2 — Modellvergleich

- Frage: welche Alternativen zu Random Forest gibt es? Gradient Boosting (XGBoost) als vielversprechendste Option identifiziert.
- `anomaly_detection_2.0.py` als separate XGBoost-Variante gebaut, um direkt vergleichbar zu bleiben (gleiche Pipeline, gleiche Splits).
- Ergebnis: XGBoost leicht besser (F1 0,59 vs. 0,57), stützt sich aber stärker auf wenige Top-Features.

## Phase 3 — Label-Problem entdeckt

- Beide Skripte mit englischen Inline-Kommentaren dokumentiert, um den Code nachvollziehbar zu machen.
- `feature_description.csv` bereitgestellt, um die Top-15-Features zu interpretieren. Auffällig: fast nur Leistungs-/Stromsensoren, kaum Temperatursensoren — untypisch für physikalische Fehler-Vorboten.
- Vermutung: Turbinen-Einstellungen könnten schon in Vorbereitung auf einen Ausfall verändert worden sein (z.B. durch Techniker).
- `status_type_id`-Bedeutung nachgeschlagen: Status 3 ("Service" — Techniker vor Ort) macht 28 % aller als "Anomalie" gelabelten Zeilen aus. Nur 24 % sind echter "Downtime".
- Zur Kontrolle geprüft, was zuerst kommt: `status_type_id` gegen die tatsächlichen Fehler-Zeitfenster (`event_start`/`event_end` in `event_info.csv`) verglichen. Ergebnis: Während der eigentlichen Vorwarnzeit bleibt der Status fast durchgehend "Normal Operation" — "Service" taucht erst danach auf. Das ursprüngliche Label maß also nicht die Vorwarnzeit, sondern erkannte primär "ist gerade ein Techniker vor Ort".

## Phase 4 — Label-Fix und korrekter Split

- Fix: neues `true_anomaly`-Label direkt aus den `event_info.csv`-Zeitfenstern abgeleitet (1 = innerhalb des Fehler-Fensters, 0 = normales Event/davor, Zeilen danach = ausgeschlossen als mehrdeutiger Nachreparatur-Zeitraum).
- Beim Retraining mit korrektem Label: Trainingsdaten enthielten 0 echte Anomalien, weil jedes Turbinen-Event komplett im `prediction`-Teil der jeweiligen Datei liegt (Datensatz-Design).
- Zwei Auswege abgewogen: (a) Turbinen-Kreuzvalidierung (ganze Turbinen zurückhalten) vs. (b) unüberwachte Novelty-Detection. Turbinen-Kreuzvalidierung gewählt, um mit dem bisherigen überwachten Ansatz vergleichbar zu bleiben.
- Nach Verifikation in separaten Testskripten (`anomaly_detection_event_labels.py`, `anomaly_detection_turbine_cv.py`) wurde die Logik in die beiden Hauptskripte übernommen; die Testskripte waren dann redundant und wurden gelöscht.

## Phase 5 — Ergebnisse mit korrigiertem Ansatz

- Random Forest: Accuracy 96 %, aber Precision/Recall nahezu 0 (0,05 % / 0,04 %) — bei starkem Klassenungleichgewicht ist hohe Accuracy hier bedeutungslos.
- XGBoost: deutlich besser als Random Forest (F1 ≈ 4,5 % vs. ≈ 0,04 %), aber insgesamt weiterhin schwach.
- Wichtiger positiver Nebenbefund: Top-Features sind jetzt physikalisch plausibel (Öl-, Lager-, Transformatortemperaturen) statt der früheren Leistungs-Artefakte — das Modell lernt jetzt zumindest die richtige Art von Signal, auch wenn die Erkennungsleistung noch gering ist.

## Phase 6 — Zeitliche Trend-Features

- Zwei neue Features pro Sensor ergänzt: Abweichung vom eigenen 24h-Durchschnitt der Turbine, und Steigung über die letzten 6 Stunden — echte Vorboten zeigen sich eher als Drift denn als einzelner Momentwert.
- Dafür beide Hauptskripte umgebaut: Trainings- und Test-Turbinen werden seither getrennt verarbeitet, da der Container nur ~7,8 GB RAM hat (davon ~2,7 GB bereits durch VS Code belegt) und alle 22 Turbinen gleichzeitig im Speicher zu halten wiederholt zu Abstürzen führte.
- Ergebnis: F1 Random Forest 0,0004 → 0,031, XGBoost 0,045 → 0,063 — spürbare Verbesserung, aber Precision weiterhin viel zu niedrig für echten Praxiseinsatz (über 90 % Fehlalarme).

## Phase 7 — Wechsel zu unüberwachtem Ansatz (Isolation Forest)

- `anomaly_detection_isolation_forest.py`: pro Turbine ein eigenes Modell, trainiert nur auf deren eigenen "train"-Zeilen (angenommen normal), bewertet auf den "prediction"-Zeilen — näher am ursprünglichen Datensatz-Design, keine Verallgemeinerung über unterschiedlich kalibrierte Turbinen mehr nötig.
- Ergebnis: Precision springt auf 41 % (vs. 5 % beim besten überwachten Modell) — deutlich weniger Fehlalarme.
- `contamination`-Parameter (erwartete Anomalie-Rate) durchgetestet: Precision bleibt über alle Werte konstant bei ~40–46 %, Recall steigt mit höherem Wert deutlich (7 % bei "auto" bis 59 % beim sklearn-Maximum von 0,5).
- Auf Vorschlag des Autors zusätzlich den F2-Score eingeführt (Recall 4× stärker gewichtet als Precision), da ein verpasster Fehler (Turbine bleibt fälschlich im Wind) schlimmer ist als ein unnötiger Fehlalarm (Turbine wird unnötig aus dem Wind gedreht). Bestes Ergebnis: `contamination=0,5` → Precision 40 %, Recall 59 %, F1 0,48, F2 0,54.

## Phase 8 — Power-Curve-Deviation-Feature (negatives Ergebnis)

- Idee: Abweichung der tatsächlichen Leistung von der erwarteten Leistungskurve (isotonische Regression Windgeschwindigkeit → Leistung, nur auf Trainingsdaten gefittet) als zusätzliches Feature, um "wenig Wind" von "zu wenig Leistung trotz genug Wind" zu trennen.
- Ergebnis: keine messbare Verbesserung der Precision.
- Zur Erklärung eine Visualisierung erstellt (`plot_power_curve.py` → `analysis_output/power_curve_deviation_examples.png`): Sie zeigt, dass die meisten als "Anomalie" gelabelten Punkte leistungsmäßig fast exakt auf der erwarteten Kurve liegen — nur eine Handvoll Punkte kurz vor dem eigentlichen Ausfall weicht sichtbar ab. Die mehrtägige Vorwarnzeit zeigt sich also kaum in der Leistungsabgabe selbst.

## Phase 9 — Feature-Auswahl verschlankt: nur Temperatursensoren

- Feature-Set auf die Temperatursensoren reduziert (dynamisch aus `feature_description.csv` gefiltert, wo die Beschreibung "temperature" enthält) plus `power_curve_deviation` — 76 statt 169 Features.
- Begründung: Isolation Forest wählt zufällige Achsen für seine Splits; viele der bisherigen, für Fehlererkennung irrelevanten Sensoren (Wind, Elektrik, Winkel) könnten das Signal verdünnt haben — und Temperatursensoren waren in jedem bisherigen überwachten Modell die mit Abstand wichtigsten Features.
- Ergebnis: deutliche Verbesserung im "auto"-Modus (F1 0,115 → 0,203, Precision 40 % → 47 %), moderate Verbesserung bei höheren Contamination-Werten.

## Einordnung: Art der Fehler und Kausalität zum Wetter

- Autor fragte, ob sich vorhersagen ließe, wann Wetter (insbesondere Windgeschwindigkeit) zu Fehlern führen könnte.
- Eingeordnet: Die vier Fehlerarten in diesem Datensatz (Generatorlager, Getriebe, Hydraulik, Transformator) sind überwiegend klassische Verschleiß-/Alterungsschäden, keine akuten, wetterbedingten Ereignisse (z.B. Sturmschäden). Ein kurzfristiger kausaler Zusammenhang mit Windextremen ist daher eher unwahrscheinlich, auch wenn Windturbulenz langfristig zur mechanischen Ermüdung von Lagern/Getriebe beitragen kann - mit nur 11 Fehlerereignissen im Datensatz ließe sich das ohnehin kaum statistisch sauber belegen.
- Zudem enthält der Datensatz nur vergangene Messwerte, keine Wettervorhersagen - eine echte "Vorhersage, wann Wetter zu Fehlern führt" ist damit nicht möglich, nur die Nutzung von Windturbulenz als zusätzlichem Risikofaktor für bereits laufende Fehlerentwicklung.
- Als Kompromiss `wind_speed_3_std` (Windböigkeit innerhalb jedes 10-Minuten-Fensters) zusätzlich zum Temperatur-Feature-Set ergänzt und getestet.
- Ergebnis: leichte Verschlechterung statt Verbesserung (F1 im "auto"-Modus 0,203 → 0,154, Precision 47 % → 41 %) — bestätigt die Einschätzung, dass kurzfristige Windböigkeit für diese Verschleißschäden kein hilfreicher Vorbote ist, sondern eher zusätzliches Rauschen. Feature-Set wieder auf reine Temperatursensoren zurückgesetzt.

## Phase 10 — Z-Score-Normalisierung pro Turbine (kein Effekt, aber lehrreich)

- Idee: Sensorwerte pro Turbine auf Durchschnitt 0 / Standardabweichung 1 normalisieren (`StandardScaler`, nur auf `train`-Zeilen gefittet), damit kein Feature durch seinen Zahlenbereich zufällig bevorzugt wird.
- Ergebnis: Zahlen exakt identisch zu vorher (bis auf letzte Nachkommastelle).
- Erklärung: Isolation Forest wählt bei jedem Split ein Feature zufällig (unabhängig von dessen Wertebereich) und einen Schwellenwert zufällig zwischen dessen Minimum und Maximum. Eine Z-Score-Normalisierung ist eine reine lineare Umskalierung pro Feature und ändert daher nichts an der relativen Position der Werte — die entstehende Baumstruktur ist mathematisch identisch. Isolation Forest ist von Natur aus unempfindlich gegenüber dieser Art Skalierungsproblem.
- Für andere Verfahren (Local Outlier Factor, One-Class SVM), die auf tatsächlichen Distanzen zwischen Punkten basieren, würde Normalisierung dagegen sehr wohl etwas bringen.

## Phase 11 — Vorlaufzeit-Kennzahl (Lead Time) und F2 als Hauptkriterium

- Autor fragte, ob man die Vorhersage 30 Minuten in die Zukunft verschieben könnte und ob das die Precision "verhauen" würde. Eingeordnet: Da die Fehler-Zeitfenster mehrere Tage lang sind, würde eine 30-Minuten-Verschiebung kaum etwas ändern - die eigentlich interessante Frage ist stattdessen, wie viel Vorlaufzeit *vor* dem offiziellen Fensterbeginn (`event_start`) erreicht wird.
- Umgesetzt: neue Kennzahl `lead_time_minutes` pro Turbine = Zeit zwischen dem ersten vom Modell markierten Punkt und `event_start` (positiv = echte Vorwarnung vor dem offiziellen Fensterbeginn). Dauerhaft in die Pipeline eingebaut (nicht nur einmalig berechnet), erscheint ab sofort bei jedem Lauf automatisch in den Per-Turbine-CSVs sowie als `early_warning_turbines`/`median_lead_time_minutes` in der Sweep-Übersicht.
- Auf Wunsch des Autors zusätzlich F2 (statt Precision oder F1) als Hauptbewertungskriterium festgelegt: Die Ergebnistabelle wird jetzt nach F2 sortiert, und der beste Wert wird am Ende automatisch hervorgehoben.
- Ergebnis (Feature-Set "nur Temperatur", bestes F2 bei `contamination=0,5`): F2 = 0,543, Precision 40,4 %, Recall 59,4 % - **7 von 11 echten Fehler-Turbinen wurden vor dem offiziellen Fensterbeginn markiert, mittlere Vorlaufzeit 2.880 Minuten (48 Stunden / 2 Tage)**. Das ist die bisher aussagekräftigste Kennzahl für den praktischen Nutzen des Systems.

## Offene Punkte / mögliche nächste Schritte

- Entscheidungsschwelle (statt festem 0,5-Cutoff) anpassen, um Precision gegen Recall zu tauschen — teilweise bereits durch den `contamination`-Sweep abgedeckt.
- Anderes distanzbasiertes Verfahren (Local Outlier Factor, One-Class SVM) testen, wo Z-Score-Normalisierung tatsächlich einen Unterschied machen würde.
- Weitere Feature-Auswahl-Experimente (z.B. nur die stärksten Einzelsensoren statt aller Temperaturen).
- Zeitliche Glättung/Bestätigung (mehrere aufeinanderfolgende auffällige Punkte statt Einzelpunkt-Alarm) zur weiteren Reduktion von Fehlalarmen.


Struktur von anomaly_detection_isolation_forest.py (389 Zeilen) mit Zeilenverweisen:

Konfiguration (Zeilen 1–86)

1–18: Modul-Docstring — Kurzbeschreibung des Ansatzes, Verweis auf anomaly_detection_journal.md.
20–31: Imports.
34–37: Pfad-Konstanten (CLEANED_DATA_DIR, EVENT_INFO_PATH, FEATURE_DESCRIPTION_PATH, OUTPUT_DIR).
39–51: Spalten-Konstanten (Zeitstempel, Status, Wind, Leistung, Rotor-/Generatordrehzahl, Pitch).
53–55: Fenstergrößen für Trend-Features (TREND_WINDOW_24H, TREND_WINDOW_SLOPE_HOURS).
57–58: Modell-Hyperparameter (N_ESTIMATORS, RANDOM_STATE).
60–69: FEATURE_SET + FEATURE_SET_EXTRA_COLS-Dict (welche Sensoren das Modell sieht).
71–73: Z_SCORE_NORMALIZE-Schalter.
75–79: CONTAMINATION_VALUES (die Sweep-Werte).
81–83: FBETA_WEIGHT (β=√2).
85–86: TOP_N_ALARM_FEATURES.

TurbineData (Zeilen 89–95)

NamedTuple-Container für die aufbereiteten Daten einer Turbine.

Feature-Engineering (Zeilen 98–138)

98–103: get_temperature_avg_columns — Temperatursensoren aus feature_description.csv.
106–115: add_power_curve_deviation — Leistungskurven-Abweichung.
117–126: add_gear_ratio_deviation — Getriebeverhältnis-Abweichung.
128–138: add_trend_features — 24h-Abweichung + 6h-Steigung pro Sensor.

Daten laden & aufbereiten (Zeilen 140–226)

140–166: load_and_label_turbine — CSV laden, Features anhängen, Zeilen labeln.
169–178: evaluate — accuracy/precision/recall/f1/fbeta berechnen.
181–226: prepare_turbine_data — alle Turbinen einmalig laden, Feature-Auswahl, Median-Fill, Normalisierung, Liste von TurbineData zurückgeben.

Modell & Bewertung (Zeilen 228–308)

228–236: compute_lead_time_minutes — Vorlaufzeit-Kennzahl.
240–248: fit_predict — IsolationForest trainieren + vorhersagen (gemeinsame Stelle für Sweep und Erklärung).
251–278: explain_alarm_features — Top-Features pro Alarm (Erklärbarkeit).
281–308: run_sweep — ein contamination-Wert über alle Turbinen laufen lassen, Metriken + Vorlaufzeit-Zusammenfassung.

Ablaufsteuerung (Zeilen 311–389)

311–320: Events laden, CSV-Pfade sammeln.
322–328: Feature-Set gemäß FEATURE_SET bestimmen.
330–332: prepare_turbine_data aufrufen.
334–348: Sweep über CONTAMINATION_VALUES, Per-Turbine-CSVs schreiben.
350–370: Sweep-Ergebnisse sortieren/speichern, besten Wert ausgeben.
372–384: explain_alarm_features für die beste contamination ausführen und speichern.
388–389: if __name__ == "__main__": main().