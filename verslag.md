# Schaakstukken herkennen met een neuraal netwerk

**Vak:** Computer Vision
**Datum:** maart 2026

---

## Wat is het doel?

Het doel van dit project is om schaakstukken automatisch te herkennen op een willekeurige foto of schermopname van een schaakbord. Het systeem moet de volledige bordpositie kunnen uitlezen: welk stuk staat op welke cel.

Er zijn 12 klassen die herkend moeten worden:

| Zwart | Wit |
|---|---|
| black-pawn | white-pawn |
| black-rook | white-rook |
| black-knight | white-knight |
| black-bishop | white-bishop |
| black-queen | white-queen |
| black-king | white-king |

---

## De dataset

Er worden twee datasets gecombineerd, beide afkomstig van Roboflow. Elke afbeelding heeft een CSV-bestand met de klasse en de coordinaten van elk stuk als bounding box. Beide datasets zijn gelicenseerd onder CC BY 4.0.

| Dataset | Afbeeldingen | Link |
|---|---|---|
| Chess.com (set 1) | 167 | https://universe.roboflow.com/ml-ki9ku/chess.com |
| Chess.com Pieces (set 2) | 183 | https://universe.roboflow.com/chess-pieces-8qwqx/chess.com-pieces |

Beide sets zijn voor export al geresized naar 640x640 pixels via Roboflow. De annotaties zijn in Tensorflow Object Detection formaat (CSV met xmin, ymin, xmax, ymax per stuk).

Een structureel probleem in de data is klasse-imbalans: pawns komen op een schaakbord standaard 16 keer voor, koningen slechts 2 keer. Dit zorgt er later voor dat het model bij twijfel richting pawn trekt.

De dataset is overwegend gemaakt in een specifieke cartoon-stijl. Dit heeft gevolgen voor generalisatie naar andere stijlen zoals chess.com, waar de stukken er anders uitzien.

---

## Fase 1: grid-gebaseerde detectie (runs 1-14)

### Hoe werkt het?

De eerste aanpak stuurt de volledige bordafbeelding in een keer door het netwerk, gebaseerd op hetzelfde principe als YOLO. Het netwerk verdeelt het beeld in een raster en voorspelt per cel of er een stuk staat, waar precies, en welk type.

De outputtensor heeft de vorm `(20, 20, 17)`. Per cel zijn dat 17 waarden:

- x en y: offset van het middelpunt binnen de cel (0 tot 1)
- w en h: breedte en hoogte als fractie van het beeld
- confidence: zekerheid dat er een stuk in deze cel staat
- 12 klassewaarden (one-hot)

### Het model (grid-detector)

```
Input: 640x640x3

Block 1  640x640  -> Conv2D 32  + BatchNorm + LeakyReLU -> MaxPool -> 320x320
Block 2  320x320  -> Conv2D 64  + BatchNorm + LeakyReLU -> MaxPool -> 160x160
Block 3  160x160  -> Conv2D 128 + Residual               -> MaxPool -> 80x80
Block 4   80x80   -> Conv2D 256 + Residual               -> MaxPool -> 40x40
Block 5   40x40   -> Conv2D 512 + BatchNorm + LeakyReLU  -> MaxPool -> 20x20

Detection head:
           20x20  -> Conv2D 256 (1x1) + Dropout 0.3
           20x20  -> Conv2D 17  (1x1) + sigmoid
Output:    20x20x17
```

Residual blokken in blocks 3 en 4 voegen de input op bij de output via een skip-connection. Dit voorkomt dat het gradient-signaal wegsterft in diepere lagen. Een residual blok bestaat intern uit een 1x1 Conv gevolgd door een 3x3 Conv, beide met BatchNorm en LeakyReLU.

BatchNorm na elke Conv laag normaliseert de activaties per batch zodat ze gemiddeld rond nul liggen. Dit stabiliseert de training en maakt het model minder gevoelig voor de initialisatie van de gewichten.

LeakyReLU geeft bij negatieve inputwaarden een kleine waarde terug (0.1 x input) in plaats van nul. Dit voorkomt dat neuronen permanent ophouden met bijdragen.

De detectiehead gebruikt geen Flatten of Dense lagen. Een 1x1 Conv behoudt de ruimtelijke structuur zodat elke uitvoercel van het 20x20 raster direct overeenkomt met een specifieke regio in het inputbeeld.

### Verliesfunctie

Standaard verliesfuncties zijn niet geschikt voor objectdetectie omdat ze geen locatiecomponent bevatten. Er is een custom loss gebouwd met drie termen:

| Term | Berekend op | Weging |
|---|---|---|
| Box loss MSE op x, y, w, h | Alleen gevulde cellen | w en h x5 |
| Confidence loss | Gevulde cellen x10, lege cellen x0.5 | Asymmetrisch |
| Class loss MSE op klassewaarden | Alleen gevulde cellen | Per class weight |

De asymmetrische confidence-weging is nodig omdat bij een 20x20 raster de meeste van de 400 cellen leeg zijn. Zonder correctie domineert het signaal "er staat niets" de training.

### Evaluatie

De hoofdmetric is mAP (mean Average Precision) bij IoU-drempel 0.3. Een detectie telt alleen als correct als zowel het label als de boxpositie kloppen. IoU meet de overlap tussen de voorspelde box en de werkelijke box.

---

### Trainingscurve (run 14)

![Training run 14](outputs/plots/run14/training_run14.png)

> De loss daalt stabiel voor zowel train als validatie. De grote piek in validatieloss rond epoch 40 is een tijdelijke instabiliteit waarna het model herstelt. De accuracy is geen betrouwbare metric bij een grid-model omdat de meeste cellen leeg zijn en correct als leeg worden voorspeld.

---

### mAP per klasse (run 14)

![mAP run 14](outputs/plots/run14/map_run14.png)

> mAP van 0.975 op de testset. Bijna alle klassen scoren boven 0.95. black-bishop is de laagste klasse maar scoort nog steeds 0.83. Dit is het eindresultaat na 14 runs van architectuur- en encoding-verbeteringen.

---

### Confusion matrix (run 14)

![Confusion matrix run 14](outputs/plots/run14/confusion_matrix_run14.png)

> De diagonaal is sterk voor alle klassen. black-bishop heeft 4 foute voorspellingen, alle als black-pawn. Alle andere klassen hebben nul of een fout.

---

### Voorspellingen op testset (run 14)

![Predictions run 14](outputs/plots/run14/predictions_run14.png)

> Links: bijna alle stukken worden correct gedetecteerd en gelabeld. De bounding boxes zitten strak om de stukken. Rechts: het model detecteert alleen de bovenste rij stukken en mist de rest. Dit is een ander bord dan de trainingsdata.

---

### Test op chess.com screenshot (run 11)

![Test1 predicted run 11](outputs/plots/run11/test1_predicted.png)

> Een echte chess.com screenshot door het grid-model. De meeste stukken worden correct gedetecteerd. De black-bishop linksboven wordt als black-pawn geclassificeerd. De blauwe UI-knop van chess.com wordt ook als stuk gedetecteerd. Dit illustreert het generalisatieprobleem: de visuele stijl van chess.com wijkt af van de trainingsdata.

---

### Samenvatting fase 1

Na 14 runs is mAP 0.975 bereikt op de testset. Het systeem werkt goed op de trainingsstijl. Het enige hardnekkige probleem is dat black-bishop wordt verward met black-pawn, zowel op de testset als op chess.com screenshots. Pogingen om dit te fixen via class weights, augmentatie en een pretrained backbone hadden geen blijvend effect.

De conclusie: een model dat tegelijkertijd positie en klasse bepaalt heeft meer moeite met subtiele klasseonderscheidingen dan een model dat zich uitsluitend op classificatie richt. Dit was de aanleiding voor fase 2.

---

## Fase 2: cel-gebaseerde classificatie (runs 15-18)

### Waarom een andere aanpak?

Het grid-model doet twee dingen tegelijk: bepalen waar een stuk staat en bepalen welk stuk het is. Door deze stappen los te trekken kan een dedicated classifier zich volledig richten op het onderscheid tussen de 12 klassen zonder rekening te houden met bounding boxes of gridlogica.

De aanpak in drie stappen:

1. Detecteer en crop het bord via OpenCV
2. Bepaal per cel of er een stuk aanwezig is via edge- en contouranalyse
3. Classificeer alleen de bezette cellen via een CNN

### Stap 1: board detection

`board_detector.py` detecteert het schaakbord via Canny edge detection en contouranalyse. Canny markeert scherpe kleurovergangen. De grootste vierhoekige contour is het bord. `getPerspectiveTransform` trekt het bord recht naar een vlak 512x512 vierkant.

![Gecropte board](classificator/outputs/plots/run17/predict_board_run17.png)

> Het perspectief-gecorrigeerde bord na board detection. De transformatie trekt het bord recht zodat alle 64 cellen exact even groot zijn na de splitsing, ook als de screenshot licht scheef is.

---

### Stap 2: piece detection per cel

Voordat de classifier een cel ziet, wordt visueel gecontroleerd of er iets aanwezig is. Dit voorkomt dat elke lege cel ook een stukclassificatie krijgt en zo false positives produceert.

Twee onafhankelijke checks:

**Edge density:** het percentage Canny-edgepixels per cel. Lege cellen zijn visueel vlak. Stukken hebben contouren en interne textuur. Drempel: meer dan 6% edgepixels.

**Adaptive threshold blob:** in het centrale 75% van de cel wordt een adaptieve threshold toegepast. Als een blob groter is dan 4% van het centrumoppervlak telt dat als stuk. Dit vangt donkere stukken op donkere cellen die weinig Canny-edges geven.

![Cel-detectie grid](classificator/outputs/plots/run17/predict_cells_run17.png)

> Alle 64 cellen als 8x8 grid. Groene rand: cel gemarkeerd als bezet en doorgestuurd naar de classifier. Rode rand: cel gemarkeerd als leeg en overgeslagen. Te veel groene cellen op lege velden zou betekenen dat de drempelwaarden te laag zijn.

![Canny edges per cel](classificator/outputs/plots/run17/predict_cells_edges_run17.png)

> De Canny edge-afbeelding per cel zoals het algoritme die intern ziet. Cellen met een stuk tonen witte lijnen langs de contouren. Lege cellen zijn vrijwel geheel zwart. De gekleurde rand geeft het eindoordeel van de piece-detection stap.

---

### Stap 3: het classifier model

Het model neemt een 80x80 patch als input en geeft een softmax-verdeling over 12 klassen terug.

```
Input: 80x80x3

Block 1  80x80  -> Conv2D 32  (3x3, no bias) + LeakyReLU(0.1) -> MaxPool -> 40x40
Block 2  40x40  -> Conv2D 64  (3x3, no bias) + LeakyReLU(0.1) -> MaxPool -> 20x20
Block 3  20x20  -> Conv2D 128 (3x3, no bias) + LeakyReLU(0.1) -> MaxPool -> 10x10

Flatten -> 11.520 waarden
Dropout 0.3
Dense 256 + LeakyReLU(0.1)
Dense 12  + softmax

Output: 12 klassewaarden die optellen tot 1.0
```

Waarom geen residual blokken of BatchNorm zoals in de grid-detector: de input is slechts 80x80 en de taak is eenvoudiger (alleen classificatie, geen lokalisatie). Een eenvoudiger model is sneller te trainen en minder gevoelig voor overfitting op de beperkte hoeveelheid patches.

Waarom `use_bias=False` op de Conv lagen: bij LeakyReLU zonder BatchNorm is de bias-term redundant omdat LeakyReLU niet volledig nul wordt voor negatieve waarden. Het weglaten van de bias verkleint het model licht.

### Training van de classifier

| Instelling | Waarde | Reden |
|---|---|---|
| Optimizer | Adam (lr 0.001) | Standaard startpunt voor classificatie |
| Loss | sparse categorical crossentropy | Labels zijn integers, niet one-hot |
| PATCH_SIZE | 80 | Meer detail per stuk dan bij 64 |
| BATCH_SIZE | 16 | Kleinere batches geven iets ruisigere gradients wat generalisatie licht verbetert |
| EPOCHS | 150 | Met ModelCheckpoint wordt sowieso het beste moment bewaard |
| ModelCheckpoint | op val_accuracy | Slaat het beste model op, niet het laatste |
| Class weights | balanced, automatisch berekend | Corrigeert voor imbalans in patchaantal per klasse |

---

### Confusion matrix classifier (run 17)

![Confusion matrix run 17](classificator/outputs/plots/run17/confusion_matrix_run17.png)

> De diagonaal is sterk voor vrijwel alle klassen. black-bishop heeft 3 foute voorspellingen als black-pawn. Alle overige klassen hebben nul of een fout. De algehele accuraatheid op de testset is hoog.

---

### Eindresultaat op chess.com screenshot (run 17)

![Classified run 17](classificator/outputs/plots/run17/predict_classified_run17.png)

> Het eindresultaat van de volledige cel-classifier pipeline op een chess.com screenshot. Per bezette cel staat de 2-lettercode van het voorspelde stuk en de bijbehorende confidence. De black-bishop linksboven (a8) wordt nog steeds als bp (black-pawn) geclassificeerd met 0.80 confidence. De chess.com UI-knop wordt ook als stuk opgepikt door de piece-detector.

---

### Vergelijking grid-detector vs cel-classifier

| Eigenschap | Grid-detector (fase 1) | Cel-classifier (fase 2) |
|---|---|---|
| Inputformaat | Volledig bord 640x640 | Losse cel 80x80 |
| Outputformaat | 20x20x17 tensor | 12 klassewaarden per cel |
| Bounding boxes | Ja, voorspeld door model | Nee, cellen zijn vast |
| Localisatie | Door model geleerd | Mechanisch (bord / 8) |
| Board detection nodig | Nee | Ja, via OpenCV |
| False positives lege cellen | Mogelijk | Gefilterd door piece-detection |
| mAP testset | 0.975 | Niet van toepassing |
| Black-bishop op chess.com | Fout | Nog steeds fout |

---

## Resterend probleem: black-bishop vs black-pawn

Beide systemen maken dezelfde fout op chess.com: de black-bishop wordt als black-pawn geclassificeerd. De oorzaak is niet de architectuur of de hyperparameters.

**Visuele stijlkloof:** de trainingsdata bevat stukken in een cartoon-stijl met overdreven duidelijke vormen. Op chess.com heeft de black-bishop een subtieler silhouet. Beide stukken zijn donkere vormen op een donkere of lichte achtergrond en lijken in die stijl meer op elkaar dan in de trainingsdata.

Dit verklaart ook waarom white-bishop geen probleem geeft: witte stukken hebben een kruis bovenop dat in elke stijl duidelijk zichtbaar is. Het onderscheid is robuuster dan bij donkere silhouetten.

**Oplossing:** 15-20 chess.com stijl black-bishop patches handmatig uitknippen en toevoegen aan de trainingsdata. Het model heeft die visuele stijl dan gezien tijdens training en kan het patroon herkennen.

---

## Samenvatting

Dit project bouwt een systeem dat schaakstukken herkent op willekeurige schermopnames van een schaakbord. Er zijn twee aanpakken geimplementeerd en vergeleken.

Fase 1 implementeert een YOLO-achtige grid-detector met een 5-blok CNN (640x640 input, residual blokken, BatchNorm, LeakyReLU) die per 20x20 gridcel positie en klasse gelijktijdig voorspelt. Na 14 runs is mAP 0.975 bereikt op de testset via iteratieve verbeteringen aan de architectuur, verliesfunctie en coordinatenencoding.

Fase 2 knipt het probleem op in losse stappen: board detection via OpenCV perspectieftransformatie, piece-detection via Canny edge density en adaptive threshold, en een aparte CNN-classifier per cel (80x80 input, 3 Conv blokken, LeakyReLU, softmax). Dit maakt het systeem eenvoudiger te debuggen, vermindert false positives op lege cellen en laat een dedicated model zich volledig richten op het klasseonderscheid.

Het black-bishop probleem blijft in beide fases bestaan en is toe te schrijven aan het visuele stijlverschil tussen de trainingsdata en chess.com, niet aan de modelarchitectuur.
