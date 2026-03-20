# Logboek — Chess Piece Object Detection CNN

---

## Dag 1 — 19/03/26

### config.py
Centrale plek voor alle paden, CLASS_MAP, image grootte en hyperparameters. Beide datasets hebben een train/valid/test split via een `SPLITS` dict met `os.path.join`. `RUN_ID` handmatig ophogen voor elke nieuwe run. Plots worden opgeslagen in `outputs/plots/run{RUN_ID}/`. Image size is 224×224. `GRID_S = 7` bepaalt hoe fijn het grid is.

### data_loader.py
Laadt volledige bordimages met alle annotaties per image. `load_images()` schaalt de bounding box coördinaten mee na het resizen naar 224×224. `plot_bbox_verification()` toont de boxes op de geresizede images zodat je kan controleren of de schaling klopt.

### preprocessing.py
Normaliseert pixels naar [0, 1]. Encodeert annotaties naar een grid target tensor van vorm `(GRID_S, GRID_S, 5 + NUM_CLASSES)` per image. Per grid cel: `[x_offset, y_offset, w_norm, h_norm, confidence, ...one-hot class...]`. De cel wordt bepaald door waar het middelpunt van de bounding box valt.

### train.py
Compileert het model met Adam optimizer en een custom detection loss. De loss bestaat uit drie delen:

| Deel | Wat het doet |
|------|-------------|
| Box loss | MSE op x, y, w, h — alleen voor cellen met een object |
| Confidence loss | MSE op confidence voor cellen met én zonder object, lege cellen weggewogen met factor 0.5 |
| Class loss | MSE op class probabiliteiten — alleen voor cellen met een object |

Cellen zonder object krijgen een lagere penalty zodat het model niet overspoeld wordt door de grote hoeveelheid lege cellen. Slaat model op als `outputs/models/model_run{RUN_ID}.h5` en de loss curve als `outputs/plots/run1/loss_run1.png`.

### Model
Input: 224×224×3 → Output: `(7, 7, 17)` — één voorspelling per grid cel.

| Laag | Waarom |
|------|--------|
| Conv2D 32 filters | Leert basale vormen en randen |
| MaxPooling 224→112 | Verkleint feature map, minder gevoelig voor kleine verschuivingen |
| Conv2D 64 filters | Leert specifiekere stukkenvormen |
| MaxPooling 112→56 | Zelfde reden |
| Conv2D 128 filters | Hogere combinaties van features |
| MaxPooling 56→28 | Zelfde reden |
| Conv2D 256 filters | Vierde blok toegevoegd omdat 224×224 meer spatiale detail heeft — meer compressie nodig voor de head |
| MaxPooling 28→14 | Zelfde reden |
| GlobalAveragePooling2D | Vervangt Flatten — middelt elke feature map naar één waarde. Veel minder parameters naar de Dense laag, minder kans op overfitten |
| Dense 512 | Combineert alle features voor de grid output |
| Dropout 0.3 | Voorkomt overfitten |
| Dense GRID_S² × 17 + Reshape | Geeft een vector per grid cel terug met box coördinaten, confidence en class scores |

### Outputbestanden

**Class distributie**  
![Class distributie run 1](outputs/plots/run1/class_distribution_train_run1.png)

**Bounding box verificatie**  
![Bbox verificatie run 1](outputs/plots/run1/bbox_verification_run1.png)

---

## Dag 2 — 19/03/26

### Training

Trainingsloop toegevoegd. De annotations worden in `train.py` omgezet naar grid tensors zodat de target encoding en de loss functie op dezelfde plek zitten.

**Loss functie**  
Standaard losses werken hier niet omdat de output drie verschillende dingen bevat die elk apart behandeld moeten worden:

| Deel | Loss | Waarom |
|------|------|--------|
| Box coördinaten (x, y, w, h) | MSE | Alleen berekend waar een object zit |
| Confidence | MSE | Dataset bevat alleen images met stukken, geen lege achtergronden — binary crossentropy voegt hier niets toe |
| Class scores | Categorical crossentropy | Alleen berekend waar een object zit |

**Checkpoints**  
Model wordt opgeslagen via `ModelCheckpoint` op basis van de laagste validatie loss. Alleen het beste model wordt bewaard als `best_model_run{RUN_ID}.h5`.

### Evaluatie & voorspellingen

`evaluate.py` — draait op de test set na training:
- Confusion matrix per class via seaborn heatmap
- MAE per box coördinaat (x, y, w, h) — alleen berekend op cellen waar een object zit

`predict.py` — toont 2 voorbeeldimages met de voorspelde bounding boxes erop getekend inclusief class naam en confidence score.

### Run 1 resultaten

![Training run 1](outputs/plots/run1/training_run1.png)
> Loss daalt snel in de eerste 3 epochs en vlakt daarna af rond 0.70. Accuracy blijft extreem laag — train ~1.4%, validatie ~0.7%. Het model leert vrijwel niets na epoch 3. De validatie loss ligt hoger dan de train loss wat wijst op overfitting.

![Confusion matrix run 1](outputs/plots/run1/confusion_matrix_run1.png)
> Het model voorspelt bijna alles als `black-pawn` of `white-pawn`. Pawns zijn oververtegenwoordigd in de dataset waardoor het model die voorkeur overneemt. Classes als `black-queen`, `white-king` en `white-bishop` worden nauwelijks correct herkend.

![MAE run 1](outputs/plots/run1/mae_run1.png)
> Positie (x ~0.185, y ~0.155) heeft een hogere fout dan afmetingen (w ~0.072, h ~0.085). Het model schat de grootte van een stuk redelijker dan de locatie — de boxes zitten dus op de verkeerde plek maar hebben wel een enigszins plausibele grootte.

![Predictions run 1](outputs/plots/run1/predictions_run1.png)
> Bounding boxes zijn kleine rode vierkantjes die de stukken niet goed bedekken. Op de tweede image wordt vrijwel niets gedetecteerd. De boxes zijn te klein en zitten verkeerd gepositioneerd. Veel voorspellingen zijn `black-pawn` of `white-pawn` ongeacht wat er werkelijk staat.

**Wat volgende run moet verbeteren**
- Class imbalance aanpakken — model bias naar pawns moet weg
- Box grootte en positie kloppen niet, de schaling in encoding of loss nakijken

---

## Dag 3 — 20/03/26

### Meer epochs

Uit run 1 bleek dat het model na epoch 3 nauwelijks meer verbeterde binnen 30 epochs. De vraag is of het model überhaupt meer kan leren als het meer tijd krijgt, of dat het probleem dieper zit. Epochs verhoogd van 30 naar 200 om dit te testen. Verder niks veranderd zodat het effect puur aan de trainingstijd toe te schrijven is.

![Training](outputs/plots/run2/training_run2.png)
> Loss schommelt de hele tijd rond 1.25 zonder structureel te dalen — het model leert niks meer. De trainloss wiggelt sterk per epoch wat wijst op een te hoge learning rate. Validatie accuracy staat bevroren op ~3%, train accuracy ~6%. Meer epochs helpt hier niet.

![Confusion matrix](outputs/plots/run2/confusion_matrix_run2.png)
> Nog steeds heavy bias naar `black-pawn` en `white-pawn`. Vergeleken met de vorige keer is de spreiding iets gelijkmatiger over andere classes maar de diagonaal is zwak — het model discrimineert slecht tussen classes.

![MAE](outputs/plots/run2/mae_run2.png)
> MAE is verslechterd ten opzichte van de vorige keer. x en y zitten nu op ~0.30 (was ~0.18 en ~0.15), w en h ook hoger. Het model plaatst boxes dus slechter dan voorheen ondanks meer training.

![Predictions](outputs/plots/run2/predictions_run2.png)
> Boxes zijn zichtbaar groter dan vorige keer en dekken soms een stuk. Maar de posities kloppen niet — er zijn grote rechthoeken die meerdere stukken bedekken in plaats van één per stuk. Veel voorspellingen zijn nog steeds pawn ongeacht wat er staat.

**Conclusie**  
Meer epochs heeft het probleem niet opgelost en op sommige punten verslechterd. De learning rate is waarschijnlijk te hoog waardoor het model niet convergeert.

### Aanpassingen

Learning rate verlaagd van `1e-3` naar `1e-4` zodat de stappen kleiner worden en de loss stabieler kan dalen. `ReduceLROnPlateau` toegevoegd — als de validatie loss 10 epochs achter elkaar niet verbetert wordt de LR automatisch gehalveerd met een minimum van `1e-6`.

Model ook iets uitgebreid: een vierde conv blok toegevoegd (256 filters) en `Flatten` vervangen door `GlobalAveragePooling2D`. `Flatten` gaf een vector van 100k+ waarden door aan de Dense laag — GAP doet dat in 256 waarden. Minder parameters, traint stabieler.

![Training](outputs/plots/run3/training_run3.png)
> Loss daalt nu wel structureel — van 1.2 naar ~0.73 voor train en ~0.76 voor validatie. De schommelingen zijn weg. Wel vlakt alles af rond epoch 50 en verbetert daarna nauwelijks meer. Validatie accuracy zakt naar bijna 0 terwijl train ~0.8% haalt — het model generaliseert slecht.

![Confusion matrix](outputs/plots/run3/confusion_matrix_run3.png)
> Bias naar pawns is er nog steeds, `white-pawn` scoort nu 82 correct maar trekt ook veel andere classes naar zich toe. Enkele classes doen het redelijker dan voorheen — `white-king` (9), `black-rook` (13), `black-knight` (10) hebben een zichtbare diagonaalwaarde. Maar het patroon is te verspreid voor betrouwbare detectie.

![MAE](outputs/plots/run3/mae_run3.png)
> Duidelijke verbetering — x ~0.118 (was ~0.185), y ~0.092 (was ~0.155), w en h zitten nu op ~0.029 en ~0.030 (was ~0.07-0.08). De boxgroottes worden erg goed geschat, de positionering is beter maar nog niet goed genoeg.

![Predictions](outputs/plots/run3/predictions_run3.png)
> Boxes zijn nu individueel per stuk in plaats van grote blokken. Op de eerste image worden meerdere stukken correct omcirkeld op de juiste plek. De tweede image detecteert niks — waarschijnlijk een te andere beeldstijl of lege bovenkant. Classes als `white-king`, `white-bishop` en `black-queen` worden nu af en toe correct voorspeld.

**Conclusie**  
Duidelijke stap vooruit — loss convergeert stabiel, MAE op w/h is bijna opgelost, boxes zitten beter. Het grootste resterende probleem is de pawn-bias en het feit dat de validatie accuracy naar nul zakt wat op overfitting wijst. Volgende stap: class imbalance aanpakken.

### mAP toegevoegd

mAP (mean Average Precision) is de standaard evaluatiemetric voor object detectie. Het combineert twee dingen tegelijk: hoe goed het model een object lokaliseert (via IoU — overlap tussen voorspelde en echte box) én hoe goed het de class correct voorspelt. Een hoge AP voor een class betekent dat het model die stukken op de juiste plek vindt met het juiste label. Dit geeft een eerlijker beeld dan alleen accuracy, omdat accuracy niet kijkt naar de kwaliteit van de bounding box.

Berekend per class bij IoU drempelwaarde van 0.5, de rode stippellijn in de grafiek is de mAP over alle classes.

![mAP](outputs/plots/run3/map_run3.png)
> mAP van 0.017 — erg laag maar het geeft wel een duidelijk beeld per class. `white-bishop` scoort het hoogst (0.068), gevolgd door `black-bishop` (0.040) en `black-knight` (0.026). Opvallend is dat `white-pawn` en `black-pawn` juist laag scoren ondanks dat het model ze het meest voorspelt — het model gokt pawn maar de box zit te vaak op de verkeerde plek waardoor de IoU onder 0.5 blijft. Classes als `black-king`, `black-queen` en `white-king` scoren nul, die worden nooit correct gelokaliseerd én geclassificeerd tegelijk.

### Verbeteringen voor volgende test

**Class weights**  
Het model gooide bijna alles als pawn omdat pawns veel vaker voorkomen in de dataset. Met class weights krijgt elke class een gewicht omgekeerd evenredig aan hoe vaak die voorkomt — zeldzame classes zoals kings en queens wegen zwaarder in de loss. Berekend als `total / (n_classes × count)`.

**BatchNormalization**  
Na elk Conv2D blok toegevoegd. Normaliseert de activaties per batch waardoor de gradienten stabieler blijven en het model consistenter traint.

![Training](outputs/plots/run4/training_run4.png)
> Grote verbetering — loss daalt stabiel van 1.2 naar ~0.54 voor train en ~0.56 voor validatie. De twee lijnen lopen dicht bij elkaar wat betekent dat het model nu generaliseert in plaats van te overfitten. Accuracy stijgt gestaag naar ~23% voor beide splits — een enorme sprong ten opzichte van de <1% van eerder.

![Confusion matrix](outputs/plots/run4/confusion_matrix_run4.png)
> De diagonaal is nu duidelijk zichtbaar voor bijna alle classes. `white-pawn` (60), `black-pawn` (54), `black-rook` (29), `white-queen` (34) en `black-bishop` (25) worden goed herkend. De pawn-bias is flink afgezwakt. Zeldzame classes als `black-king` (13) en `white-bishop` (13) worden nu ook gedetecteerd.

![MAE](outputs/plots/run4/mae_run4.png)
> x ~0.122, y ~0.109 — vergelijkbaar met de vorige keer. w en h licht verslechterd (~0.049 en ~0.053 vs ~0.029 en ~0.030). De positiefout is vrijwel gelijk maar de class weights hebben de boxgroottes iets beïnvloed.

![mAP](outputs/plots/run4/map_run4.png)
> mAP gedaald naar 0.011 (was 0.017). `black-king` scoort nu het hoogst (0.041), `white-rook` (0.026). Opvallend: betere classificatie maar lagere mAP — het model voorspelt meer classes correct maar de box locaties kloppen nog niet goed genoeg voor IoU 0.5.

![Predictions](outputs/plots/run4/predictions_run4.png)
> Duidelijk beter — veel meer classes worden correct benoemd, niet meer alleen pawns. Boxes zijn nog klein en zitten niet altijd precies op het stuk maar de labels kloppen veel vaker. Tweede image detecteert nu ook enkele stukken.

**Conclusie**  
Class weights en BatchNorm hebben het meeste effect gehad — accuracy van <1% naar ~23% en de diagonaal in de confusion matrix is eindelijk zichtbaar. De mAP is laag doordat de boxes te klein zijn en de IoU drempel van 0.5 niet gehaald wordt.

### GAP vervangen door spatiale Conv2D output

GlobalAveragePooling middelde elke feature map naar één getal — het model verloor daarmee alle informatie over waar op het bord een stuk stond. Classificatie verbeterde maar boxposities niet, want positie-informatie was al weggegooid voor de output.

Oplossing: GAP en de Dense lagen eruit. In plaats daarvan een vijfde MaxPool die de feature map van 14×14 naar 7×7 brengt — exact de grid grootte. Daarna een 1×1 Conv2D die per cel direct de box coördinaten en class scores uitgeeft. Elke cel in de output correspondeert nu rechtstreeks met een regio in de input image.

![Training](outputs/plots/run5/training_run5.png)
> Grote verbetering — loss daalt naar ~0.02 voor train en ~0.08 voor validatie. Accuracy stijgt naar ~35% train en ~39% validatie. De validatie accuracy ligt hoger dan train wat ongewoon is maar positief — het model generaliseert goed.

![Confusion matrix](outputs/plots/run5/confusion_matrix_run5.png)
> De diagonaal is nu zeer sterk — bijna alle classes worden correct geclassificeerd. `black-pawn` (123), `white-pawn` (114), `black-rook` (42), `white-queen` (44) en `white-rook` (40) scoren hoog. Verwarring tussen classes is minimaal geworden.

![MAE](outputs/plots/run5/mae_run5.png)
> x ~0.093, y ~0.078, w ~0.062, h ~0.070. Alle coördinaten vergelijkbaar — geen grote uitschieters meer. Maar de boxes zijn visueel nog steeds te klein en zitten niet goed om de stukken heen ondanks de lagere MAE.

![mAP](outputs/plots/run5/map_run5.png)
> mAP = 0.000 — vrijwel nul. Dit komt doordat de boxes te klein zijn waardoor de IoU nooit boven 0.5 komt. De classificatie is sterk verbeterd maar de box coördinaten die het model voorspelt corresponderen niet met de werkelijke afmetingen van een stuk. De `encode_targets` functie slaat de box positie op als cel-relatieve coördinaten maar de output van het model wordt geïnterpreteerd als absolute image coördinaten — dit is een mismatch in de predict code.

![Predictions](outputs/plots/run5/predictions_run5.png)
> Labels kloppen bijna allemaal — elke stuk krijgt de juiste naam met confidence ~1.00. De boxes zitten echter telkens linksboven van het stuk in plaats van eromheen. Op de tweede image worden de queens correct benoemd maar de rest mist.

**Conclusie**  
Classificatie is nu goed. Het probleem zit in hoe de voorspelde coördinaten worden omgezet naar pixels in `predict.py` — de cel-relatieve encoding wordt niet correct teruggerekend naar absolute beeldcoördinaten.