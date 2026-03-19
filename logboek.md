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

### model.py
Architectuur gewijzigd van classificatie naar object detectie.

Input: 224×224×3 → Output: `(7, 7, 17)` — één voorspelling per grid cel.

| Laag | Waarom |
|------|--------|
| Conv2D 32 filters | Leert basale vormen en randen |
| MaxPooling 224→112 | Verkleint feature map, minder gevoelig voor kleine verschuivingen |
| Conv2D 64 filters | Leert specifiekere stukkenvormen |
| MaxPooling 112→56 | Zelfde reden |
| Conv2D 128 filters | Derde blok nodig omdat 224×224 input meer spatiale detail bevat dan de oude 64×64 patches — meer lagen nodig om dat te comprimeren |
| MaxPooling 56→28 | Zelfde reden |
| Dense 1024 | Combineert alle features voor de grid output |
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