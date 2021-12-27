## Tulokset ja analyysi

### Datan käsittely
Kuva-aineisto on tarkoitus saattaa neuroverkolle sopivasti luokiteltuun muotoon, one-hot koodatuiksi vektoreiksi. ImageDataGenerator luokka tekee suuren osan esikäsittelystä, koska neuroverkot käsittelevät huonosti suuria kokonaislukuja kuvien pikselit normalisoidaan välille [0,1]. Lisäksi suoritetaan data augementaatiota eri parametreillä, jotta saadaan suurempi otos oppimisprosessille. Aineisto ladataan ImageDataGenerator-luokan avulla, joka luokittelee kuvat automaattisesti. Kuvia on yhteensä 251, jotka on jaettu 32 harmaanväriskaalan kuvan osiin, joissa on pikseleitä 256x256.


### Malli
Projektiin valitaan Sequential() malli, joka mahdollistaa neuroverkon kokoamisen kerros kerrokselta. Kuva-aineistoa käsitellään Convotional Neural Networkilla (CNN). Mallin input-kerroksen muodon tulee vastata kuvia ja niiden väriskaalaa. Convotional 2D kerroksissä määritellään filter-arvo ja filterille korkeus ja leveys. Filteri määrittää sen kuinka monia eri muotoja kuvasta tallennetaan seuraavalle kerrokselle. Stride parametri määrittään sen kuinka monen pikselin yli filteri liikkuu. Pooling-kerrokset vähentävät kuvien ulottuvuuksia, joka vähentää ylisovittamista ja neuroverkon kerroksien kokoa. Flatten() funktio muuttaa kuvat yhdeksi vektoriksi. Mallin output-kerros on kategorinenvektori eri kuvien todennäköisyyksistä softmax aktivaatiofunktio tuottaa vektorin.



### Mallin sovitus
Mallin loss-funktioksi valitaan ristientropia (cross-entropy), joka määrittää eron oikein ja ennustetun kuvien jakauman välillä. Algoritmi minimoin ristientropia arvoa. Eron ollessa nolla luokittelu onnistuu täydellisesti. Hyvässä mallissa ristientropian olisi mahdollisimman lähellä nollaa. Metrics arvioifunktion tarkkuutta. Metrics-funktiot kertovat mallin tarkkuudesta, eli kuinka hyvin malli kykenee luokittelemaan kuvia. Accurary-funktio laskee kuinka usein ennusteet vastaavat oikea kuvaluokkaa. Accuracy ilmoitetaan prosentteina. AUC-funkio (area under ROC-curve) eli pinta-ala ROC-käyrän (receiver operating characteristic curve) alapuolella kertoo oikeiden posiviitisten määrän ja väärien positiivisten luokitteluiden määrä eli kuinka usein malli luokittelee keuhkokuumeen kehkokuumeeksi ja kuinka usein ei AUC-arvon halutaan olevan mahdollisimman lähellä 1.0, jolloin luokittelu onnistuisi mallilta hyvin. Optimointi algoritmiksi valitaan Adam-algoritmi.


### Tulokset
Mallin accuracy ~ 0.37, joka ei ole erityisen hyvä. Accuracy funktioiden kehityksen plottaaminen jokaisen iteraation jälkeen toisaalta näyttää, että malli ei ole kovin hyvä. Precision arvo: kuinka monta prosenttia kuvista malli luokittelee oikein. Recall arvo: kuinka monta prosenttia kaikista luokan kuvista malli tunnisti
F1 arvo: painotettu keskiarvo precision ja recall arvoista kuvastaa koko mallin ennuste kykyä hyvin, silloin kun ainesto ei ole tasa-painoinen, mitä lähempänä 1 sen parempi malli on.

![kuva lossfunktiosta](/kuvat/classification.png)


![kuva classificaatio tablesta](/kuvat/lossfunction.png)

### Mallin ennustettavuuden parantaminen
Malliin hyperparametreja muokattiin manuaalisesti. Ensin yrittämällä lisätä mallin oppimismäärää nostamalla iteraatiokierroksi, tämä paransi mallin ennustettavuutta. Oppimisnopeuden nostaminen ei lisännyt mallin ennustettavuutta. Malliin yritettiin myös lisätä kerroksia, mutta malli ei lisäkerroksilla parantunut.
