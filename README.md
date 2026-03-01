# Struttura e funzionalità principali dell’applicazione
Questo programma implementa un’interfaccia grafica in Tkinter finalizzata all’analisi di immagini e alla visualizzazione delle sue carattereistiche.
L’applicazione consente di caricare un dataset di immagini, visualizzare per ciascun elemento sia l’immagine originale sia la corrispondente maschera binaria ottenuta tramite threshold RGB, e modificare in modo interattivo i parametri di sogliatura. 
Inoltre, permette l’estrazione di feature geometriche e testurali, il salvataggio dei descrittori calcolati direttamente nel CSV di input e l’elaborazione automatica dell’intero dataset.
Il programma integra anche una sezione dedicata all’analisi esplorativa dei dati, basata sulla Principal Component Analysis (PCA) delle feature estratte e sull’applicazione dell’algoritmo K-Means ai punteggi PCA, al fine di evidenziare eventuali strutture o raggruppamenti presenti nei dati. 

Nel complesso, lo strumento è pensato per supportare la costruzione, l’organizzazione e una prima analisi quantitativa di dataset di feature ricavate da immagini.

## Estrazione delle feature
Come anticipato, il programma consente di estrarre dalle immagini un insieme di feature e di salvarle dinamicamente nel file CSV associato al dataset. I descrittori calcolati appartengono a due categorie principali: **feature geometriche** e **feature testurali**. Entrambe vengono calcolate esclusivamente sulla regione dell’immagine selezionata dopo l’applicazione del threshold, cioè sulla porzione che rimane all’interno della maschera binaria ottenuta dalla sogliatura. Le feature estratte possono essere visualizzate temporaneamente all’interno dell’interfaccia e, successivamente, salvate nel file CSV sia per la singola immagine corrente sia per l’intero dataset.
Le feature geometriche estratte sono molteplici e descrivono sia la dimensione sia la forma dell’oggetto segmentato (regione contenuta nella maschera binaria). In particolare, il vettore dei descrittori include:
- height: altezza del bounding box;
- width: larghezza del bounding box;
- area: area della regione segmentata;
- equivalent_diameter: diametro equivalente, cioè il diametro del cerchio avente la stessa area della regione;
- aspect_ratio: rapporto tra larghezza e altezza del bounding box
- extent: rapporto tra area della regione e area del bounding box;
- solidity: rapporto tra area della regione e area del suo inviluppo convesso , utile per quantificare concavità e irregolarità;
A questi si aggiungono i 7 momenti invarianti di [Hu](https://it.wikipedia.org/wiki/Momento_(elaborazione_delle_immagini)) (hu1–hu7), che costituiscono descrittori di forma invarianti a traslazione, rotazione e (in prima approssimazione) scala, e consentono quindi un confronto più robusto tra oggetti con orientamento diverso.

Le feature testurali descrivono la distribuzione dei livelli di intensità (e quindi la “granularità” e l’eterogeneità) della regione segmentata. Nel programma sono calcolate sulla ROI definita dalla maschera binaria e includono:
- mean_color: media dei valori di intensità nella ROI (per canale);
- variance_color: varianza dei valori di intensità nella ROI (per canale);

A queste si aggiungono le feature di [Haralick](https://mahotas.readthedocs.io/en/latest/)  derivate dalla Gray-Level Co-occurrence Matrix (GLCM), per la maggior parte calcolate tramite la libreria Mahotas: Angular Second Moment (Energy), Contrast, Correlation, Variance, Inverse Difference Moment (Homogeneity), Sum Average, Sum Variance, Sum Entropy, Entropy, Difference Variance, Difference Entropy, Information Measure of Correlation 1, Information Measure of Correlation 2.

## PCA e clustering
Una volta aperta la finestra dedicata alla visualizzazione, è possibile applicare la **PCA** al vettore delle feature estratte in precedenza. L’utente può scegliere il numero di componenti principali da calcolare e selezionare quali due componenti rappresentare nel grafico, così da visualizzarle tra loro in un piano bidimensionale. Successivamente, sui punteggi ottenuti dalla PCA, è possibile applicare un algoritmo di clustering, nello specifico il **K-Means**, al fine di suddividere i dati in gruppi. L’appartenenza di ciascun punto al relativo cluster viene quindi mostrata graficamente mediante l’uso di colori differenti.

# Come usarlo
Per utilizzare correttamente l’applicazione è necessario disporre di un file CSV strutturato in modo opportuno, contenente i percorsi delle immagini da analizzare.
Un esempio di struttura di cartella e del relativo file CSV è disponibile nella cartella (dataset_prova)[dataset_prova].
Non è necessario che tutte le immagini siano contenute nella stessa directory: è sufficiente che il file CSV includa i percorsi corretti di tutte le immagini che si desidera elaborare.
Nel caso in cui tutte le immagini siano raccolte all’interno di una singola cartella, è possibile generare automaticamente tale file mediante la funzione `directory_immagini_to_csv`. Sarà sufficiente richiamare la funzione fornendo in input il percorso della cartella contenente le immagini, così da ottenere un CSV compatibile con il programma
```bash
path="your_own_path"
directory_immagini_to_csv(path)
```
Potete trovare un esempio di cartella in 
Una volta caricato il file CSV, sarà possibile applicare il threshold all’immagine corrente; il risultato della sogliatura verrà visualizzato sotto forma di maschera binaria nel pannello posto a sinistra dell’interfaccia. Accanto ai parametri del threshold saranno inoltre presenti due valori dedicati alla definizione dei parametri di una specifica feature testurale, il Local Binary Pattern (LBP).

Nella barra principale superiore sarà presente il pulsante “Carica CSV”, che consentirà di navigare all’interno del filesystem e selezionare il file CSV di input. Nella parte centrale della stessa barra saranno invece disponibili tre pulsanti dedicati alla gestione delle feature. Il pulsante “Estrai feature” permetterà di calcolare e visualizzare le feature relative all’immagine correntemente selezionata. Il pulsante “Salva feature” consentirà di memorizzare nel file CSV le feature associate alla sola immagine corrente. Infine, il pulsante “Salva feature dataset” eseguirà l’estrazione e il salvataggio delle feature per l’intero insieme di immagini contenute nel dataset.**
nel lato destro ci sarà il bottone **visualizza PCA**.

#### Visualizza PCA
Dopo aver cliccato il pulsante, si aprirà una nuova finestra dedicata alla visualizzazione dei risultati della PCA. Tramite i menu “Asse X” e “Asse Y” sarà possibile selezionare quali componenti principali rappresentare nel grafico. Inoltre, attraverso l’apposito menu a tendina, l’utente potrà modificare il numero di componenti della PCA da calcolare. Ogni volta che questo valore viene aggiornato, sarà necessario premere nuovamente il pulsante “Calcola” per rielaborare l’analisi e aggiornare la visualizzazione.

Premendo invece il pulsante “K-means”, l’applicazione eseguirà il clustering dei dati, generando un numero di cluster pari al valore selezionato nel menu a tendina “k-cluster”. I cluster ottenuti verranno poi rappresentati nel grafico sottostante mediante una diversa colorazione dei punti, così da rendere immediatamente visibile l’appartenenza di ciascuna osservazione al relativo gruppo.

# Utilizzo tramite interprete Python

Il programma può essere eseguito direttamente a partire dal codice sorgente mediante un interprete Python. In questo caso, è innanzitutto necessario scaricare o clonare il progetto nella propria macchina e posizionarsi, tramite terminale, nella cartella principale del repository.
```bash
git clone https://github.com/GiovanniGueltrini/ObjectFeatureExtractor.git
cd ObjectFeatureExtractor
```
Una volta creato e attivato l’ambiente virtuale, sarà possibile installare tutti i pacchetti necessari mediante il file requirements.txt, eseguendo il comando:
```bash
pip install -r requirements.txt
```
Completata l’installazione delle dipendenze, il programma potrà essere avviato eseguendo il file principale dell’applicazione, ad esempio:
```bash
python Dashboard.py
```
L’interfaccia grafica verrà quindi aperta e sarà possibile utilizzare il programma caricando un file CSV compatibile con la struttura richiesta. Questa modalità è particolarmente indicata per utenti che desiderano esaminare, modificare o sviluppare ulteriormente il codice sorgente dell’applicazione.

# Utilizzo tramite file eseguibile

In alternativa, il programma può essere utilizzato tramite un file eseguibile .exe, senza la necessità di installare manualmente Python o le librerie richieste dal progetto. In questo caso, sarà sufficiente scaricare la cartella contenente l’eseguibile e avviare il programma facendo doppio clic sul file corrispondente.

Una volta aperta l’interfaccia grafica, l’utente potrà caricare il file CSV contenente i percorsi delle immagini da analizzare e utilizzare normalmente tutte le funzionalità del software, inclusi il threshold, l’estrazione delle feature, il salvataggio dei descrittori e la visualizzazione della PCA con clustering K-Means.

È tuttavia importante osservare che l’eseguibile non sostituisce i dati di input necessari al funzionamento del programma. Di conseguenza, per un utilizzo corretto sarà comunque necessario disporre del file CSV e delle immagini a esso associate. Inoltre, i percorsi riportati nel CSV dovranno essere validi anche nel computer in cui il programma viene eseguito; in caso contrario, le immagini non potranno essere caricate correttamente.
