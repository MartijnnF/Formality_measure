#load dependencies
from vosk import Model, KaldiRecognizer
import pyaudio
import json
from pydantic import FilePath
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TFAutoModel
import torch, requests, re, spacy, os
import itertools
# from gtts import gTTS
import playsound
import requests
from bs4 import BeautifulSoup
import html.parser
import logging
import parselmouth
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

holiday_form_lvl_1 = ["Hallo, mijn naam is Alex. Hoe heet u?", 
"Hoi daar! Het is superleuk om je te ontmoeten. Ben je er helemaal klaar voor een mooie vakantie naar het strand te plannen?", 
"Ik ga je een paar vragen stellen over jouw perfecte vakantie aan het strand! Zou je liever op een doe-vakantie of een relaxvakantie gaan en waarom?", 
"Dankjewel voor je antwoord! Ik zou je ook graag willen vragen welke activiteiten jij echt tof zou vinden bij een vakantie naar het strand?", 
"Op vakantie, zit je liever lekker knus dicht bij huis, of zoek je liever het avontuur ver weg van Nederland, en waarom?", 
"Vind je het handig om een gids te hebben op vakantie? Waarom wel of niet?", 
"Wat soort plaats verblijf je het liefste in als je op deze vakantie gaat en waarom?", 
"Zijn er dingen die je echt niet kunt missen bij de plek van jouw verblijf?", 
"Als jij op vakantie gaat, is de cultuur van jouw bestemming belangrijk voor je? Waarom wel of niet?", 
"We zijn jammer genoeg bijna klaar! Zou je me kunnen vertellen hoe je de ervaring met de stem assistent vond?", 
"Zijn er misschien nog wat extras waar ik rekening mee kan houden voor deze vakantie?", 
"We zijn bij het einde van de vragenlijst, ik vond dat het super goed ging! Echt heel erg bedankt dat je de tijd hiervoor hebt genomen, de onderzoeker zal je zo iets laten weten!", 
"Harstikke leuk dat je mee hebt gedaan aan dit onderzoek, heel erg bedankt!"]

holiday_form_lvl_5 = ["Hallo, mijn naam is Alex. Hoe heet u?",
"Aangenaam om met u kennis te maken. Bent u gereed om een vakantie naar het strand te boeken?",
"Ik wens u een paar vragen te stellen over de vakantie aan het strand die het meest passend is voor u. De eerste vraag luidt: Zou u mij kunnen vertellen of uw voorkeur ligt bij een meer actieve vakantie, of bij een vakantie waar u meer de rust opzoekt, en waarom uw voorkeur hier ligt?",
"De laatste vraag volgend, kunt u enkele voorbeelden geven van activiteiten die u verlangt te doen bij een vakantie naar het strand?",
"Reist u graag naar een locatie in of nabij Nederland, of beoogt u liever een meer verafgelegen locatie? Waarom heeft u deze mening?",
"Is het voor u van belang om op vakantie begeleid te worden door een gids? Waarom vind u van wel of van niet?",
"Op welke wijze van accommodatie verblijft u graag op deze vakantie en waarom?",
"Heeft u specifieke vereisten aangaande het selecteren van de locatie van uw accommodatie?",
"Is de oorspronkelijke cultuur van uw bestemming van groot belang voor u? Waarom wel of niet?",
"U nadert het einde van deze vragenlijst. Zou u kort kunnen evalueren hoe u deze interactie met de stem assistent ervaren heeft?",
"Heeft u nog specifieke wensen voor uw vakantie waar aanvullende aandacht aan besteed dient te worden?",
"Dit is het slot van de vragenlijst, naar mijn mening is alles in goede orde verlopen. Hartelijk bedankt voor de tijd die u heeft genomen, de onderzoeker zal u nu verder begeleiden.",
"Ik dank u voor uw coöperatie."]

holiday_form_lvl_adaptable = [["Hallo, mijn naam is Alex. Hoe heet u?", "Hallo, mijn naam is Alex. Hoe heet u?", "Hallo, mijn naam is Alex. Hoe heet u?", "Hallo, mijn naam is Alex. Hoe heet u?", "Hallo, mijn naam is Alex. Hoe heet u?"],
["Hoi daar! Het is superleuk om je te ontmoeten. Ben je er helemaal klaar voor een mooie vakantie naar het strand te plannen?", "Wat leuk om je te ontmoeten! Heb je er zin in om een vakantie naar het strand te plannen?", "Wat leuk om u te ontmoeten! Heeft u er zin in om een vakantie naar het strand te plannen?", "Aangenaam om u te ontmoeten. Kijkt u ernaar uit om een vakantie naar het strand te plannen?", "Aangenaam om met u kennis te maken. Bent u gereed om een vakantie naar het strand te boeken?"],
["Ik ga je een paar vragen stellen over jouw perfecte vakantie aan het strand! Zou je liever op een doe-vakantie of een relaxvakantie gaan en waarom?", "Ik stel je graag een aantal vragen over jouw ideale vakantie aan het strand. De eerste vraag is of jouw voorkeur ligt bij een actieve vakantie of een relaxvakantie, en waarom heb je deze voorkeur?", "Ik stel u graag een aantal vragen over uw ideale vakantie aan het strand. De eerste vraag is of uw voorkeur ligt bij een actieve vakantie of een relaxvakantie en waarom heeft u deze voorkeur?", "Ik zou u graag een paar vragen stellen over uw ideale vakantie aan het strand. Allereerst zou ik u willen vragen of uw voorkeur ligt bij een actieve vakantie of een vakantie waarbij u uit kunt rusten, en waarom u deze voorkeur heeft?", "Ik wens u een paar vragen te stellen over de vakantie aan het strand die het meest passend is voor u. De eerste vraag luidt: Zou u mij kunnen vertellen of uw voorkeur ligt bij een meer actieve vakantie, of bij een vakantie waar u meer de rust opzoekt, en waarom uw voorkeur hier ligt?"],
["Dankjewel voor je antwoord! Ik zou je ook graag willen vragen welke activiteiten jij echt tof zou vinden bij een vakantie naar het strand?", "Inhakend op de laatste vraag, zou ik graag willen vragen naar wat voor activiteiten je uitkijkt bij een vakantie naar het strand?", "Inhakend op de laatste vraag, zou ik graag willen vragen naar wat voor activiteiten u uitkijkt bij een vakantie naar het strand?", "Met het oog op de laatste vraag, zou ik u graag willen vragen naar wat activiteiten waarnaar u uitkijkt bij een vakantie naar het strand?", "De laatste vraag volgend, kunt u enkele voorbeelden geven van activiteiten die u verlangt te doen bij een vakantie naar het strand?"],
["Op vakantie, zit je liever lekker knus dicht bij huis, of zoek je liever het avontuur ver weg van Nederland, en waarom?", "Vind je het belangrijk dat de locatie dichtbij is, of ga je liever verder weg, en waarom?", "Vind u het belangrijk dat de locatie dichtbij is, of gaat u liever verder weg? Waarom?", "Reist u liever naar een locatie dichtbij, of gaat uw voorkeur uit naar een verdere locatie? Waarom is dat zo?", "Reist u graag naar een locatie in of nabij Nederland, of beoogt u liever een meer verafgelegen locatie? Waarom heeft u deze mening?"],
["Vind je het handig om een gids te hebben op vakantie? Waarom wel of niet?", "Vind je dat een gids belangrijk is voor een vakantie? Waarom wel of niet?", "Vind u dat een gids belangrijk is voor een vakantie? Waarom wel of niet?", "Is een gids voor u van belang op vakantie? Waarom wel of niet?", "Is het voor u van belang om op vakantie begeleid te worden door een gids? Waarom vind u van wel of van niet?"],
["Wat soort plaats verblijf je het liefste in als je op deze vakantie gaat en waarom?", "Wat voor vakantieverblijf vind je het fijnst voor deze vakantie en waarom?", "Wat voor vakantieverblijf vind u het fijnst voor deze vakantie en waarom?", "Wat voor verblijf heeft u graag op deze vakantie en waarom?", "Op welke wijze van accommodatie verblijft u graag op deze vakantie en waarom?"],
["Zijn er dingen die je echt niet kunt missen bij de plek van jouw verblijf?", "Heb je bepaalde eisen die vastzitten aan het kiezen van de plaats van jouw accommodatie?", "Heeft u bepaalde eisen die vastzitten aan het kiezen van de plaats van uw accommodatie?", "Heeft u specifieke vereisten bij het kiezen van de locatie van uw accommodatie?", "Heeft u specifieke vereisten aangaande het selecteren van de locatie van uw accommodatie?"],
["Als jij op vakantie gaat, is de cultuur van jouw bestemming belangrijk voor je? Waarom wel of niet?", "Is de lokale cultuur van jouw bestemming belangrijk voor je? Waarom wel of niet?", "Is de lokale cultuur van uw bestemming belangrijk voor u? Waarom wel of niet?", "Is de lokale cultuur van uw bestemming van belang voor u? Waarom wel of niet?", "Is de oorspronkelijke cultuur van uw bestemming van groot belang voor u? Waarom wel of niet?"],
["We zijn jammer genoeg bijna klaar! Zou je me kunnen vertellen hoe je de ervaring met de stem assistent vond?", "Wij zijn bij het laatste deel van de vragenlijst aangekomen, hoe vond je deze ervaring met de stem assistent?", "Wij zijn bij het laatste deel van de vragenlijst aangekomen, hoe vond u deze ervaring met de stem assistent?", "U heeft het laatste deel van deze vragenlijst bereikt. Zou u met mij kunnen delen hoe u deze ervaring met de stem assistent vond?", "U nadert het einde van deze vragenlijst. Zou u kort kunnen evalueren hoe u deze interactie met de stem assistent ervaren heeft?"],
["Zijn er misschien nog wat extra\'s waar ik rekening mee kan houden voor deze vakantie?", "Zijn er nog speciale wensen voor deze vakantie waar wij naar kunnen kijken?", "Zijn er nog speciale wensen voor deze vakantie waar wij op moeten letten?", "Heeft u nog bijzondere wensen voor deze vakantie waar wij aandacht aan moeten besteden?", "Heeft u nog specifieke wensen voor uw vakantie waar aanvullende aandacht aan besteed dient te worden?"],
["We zijn bij het einde van de vragenlijst, ik vond dat het super goed ging! Echt heel erg bedankt dat je de tijd hiervoor hebt genomen, de onderzoeker zal je zo iets laten weten!", "Dit is het einde van deze vragenlijst, ik vond dat her erg goed ging. Bedankt voor de tijd die je hebt genomen, de onderzoeker zal je nu verder begeleiden.", "Dit is het einde van deze vragenlijst, ik vond dat het erg goed ging. Bedankt voor de tijd die u heeft genomen, de onderzoeker zal u nu verder begeleiden.", "Dit is het einde van deze vragenlijst, naar mijn mening is het erg goed gegaan. Hartelijk bedankt voor de tijd die u heeft genomen, de onderzoeker zal u nu verder begeleiden.", "Dit is het slot van de vragenlijst, naar mijn mening is alles in goede orde verlopen. Hartelijk bedankt voor de tijd die u heeft genomen, de onderzoeker zal u nu verder begeleiden."],
["Harstikke leuk dat je mee hebt gedaan aan dit onderzoek, heel erg bedankt!", "Heel erg bedankt voor je medewerking!", "Heel erg bedankt voor uw medewerking!", "Bedankt voor uw medewerking.", "Ik dank u voor uw coöperatie."]]

formal_list_final = ['aandachtig', 'aan de hand van', 'aangaande', 'aangezien', 'aanmerkelijk', 'aanpassingen realiseren', 'aanstonds', 'aansturen', 'aan te gane', 'aantrekken', 'aanvang', 'aanvang nemen', 'een', 'aanvangen', 'aanvankelijk', 'aanvliegen', 'aanwenden', 'abusievelijk', 'accenten verleggen', 'acceptatie', 'accommodatie', 'accorderen', 'achten', 'achttal', 'een', 'activiteiten voortzetten', 'actualiseren', 'acuut', 'ad', 'additioneel', 'adequaat', 'adhesie', 'adhesie betuigen', 'ad hoc', 'adstrueren', 'advies uitbrengen', 'afdoening', 'affirmatief', 'afgezien van', 'afronden', 'aftikken', 'afvloeien', 'aldaar', 'aldra', 'al dan niet', 'aldus', 'alhoewel', 'alleszins', 'alloceren', 'alom', 'alsdan', 'alsmede', 'alsook', 'alstoen', 'alvorens', 'alwaar', 'ambivalent', 'anderszins', 'anderzijds', 'animo', 'annonce', 'anticiperen', 'appellant', 'appelleren', 'appelleren aan', 'appreciatie', 'a priori', 'autonoom', 'beduidend', 'behagen', 'behelzen', 'behoefte bestaat aan', 'behoeven', 'behoren', 'behoudens', 'behoudens indien', 'behoudens in het geval dat', 'behoudens voor zover', 'bekomen', 'belanghebbende', 'beleidsimpuls', 'beleidsintensiveringen', 'benchmarking', 'benevens', 'beogen', 'bepaald', 'het bepaalde', 'berichten', 'berispen', 'berouw', 'bescheiden', 'beslag krijgen', 'bestendigen', 'bestrijken', 'betreffende', 'betreft', 'betrekking hebben op', 'betuigen', 'bewerkstelligen', 'bezien', 'bezigen', 'bezoldigd', 'bij brief', 'bij dezen', 'bijdrage leveren', 'bij gelegenheid', 'bijgeval', 'bijgevolg', 'bij ontstentenis van', 'bij schrijven van 1 maart', 'bij wijze van', 'bilateraal', 'binnen de gemeentelijke organisatie', 'binnen het raam van onze voorwaarden', 'black spot', 'blijkens', 'borgen', 'bovengenoemde', 'bovenstaand', 'branding', 'brandstofverkooppunt', 'cq', 'capaciteit', 'casu quo', 'categorie', 'cie', 'circa', 'citymarketing', 'clausule', 'clusteren', 'cohesie', 'compatibel', 'compensatie', 'complex', 'compliceren', 'concept', 'concipiëren', 'concreet', 'conditie', 'configuratie', 'conflicteren', 'conflictsituatie', 'conform', 'congruent', 'consensus', 'consequent', 'consistent', 'consolidatie', 'constateren', 'constellatie', 'constitueren', 'constructief', 'consultatief', 'consulteren', 'continueren', 'convenant', 'coördineren', 'courant', 'creëren', 'criterium', 'cruciaal', 'cum suis', 'cumulatief', 'curieus', 'daar', 'daarenboven', 'dagtekening', 'dan wel', 'danig', 'dankzeggen', 'dat wil zeggen', 'decentraal', 'de dato', 'deductief', 'de facto', 'degeen', 'de handen ineenslaan', 'delegeren', 'de mening toegedaan zijn', 'der', 'deregulering', 'derhalve', 'derogatie', 'derven', 'derving', 'desalniettemin', 'desbetreffend', 'desiderata', 'desideratum', 'desgevraagd', 'desniettegenstaande', 'desondanks', 'destijds', 'deswege', 'dezer dagen', 'dezerzijds', 'dicta', 'dictum', 'dienaangaande', 'dienen te', 'dienovereenkomstig', 'dientengevolge', 'differentiëren', 'discontinu', 'diskwalificatie', 'discrepantie', 'distribueren', 'dit schrijven', 'diversiteit', 'doch', 'doen toekomen', 'dogmatisch', 'door middel van', 'doorgang vinden', 'doorontwikkelen', 'doorpakken', 'draagvlak', 'dralen', 'drietal', 'dusdanig', 'echter', 'edoch', 'educatief', 'een aantal', 'een dezer dagen', 'een en ander', 'een klein aantal', 'eerdergenoemde', 'eerst dan', 'effectueren', 'elders', 'elimineren', 'elkeen', 'enerzijds', 'enige', 'enigszins', 'enkel', 'entameren', 'epistel', 'equivalent', 'ergo', 'ertoe strekken', 'ervaring', 'escaleren', 'essentieel', 'evaluatie', 'evalueren', 'evenmin', 'evenwel', 'evenzeer', 'evenzo', 'evident', 'ex', 'exceptioneel', 'excerperen', 'excessief', 'ex nunc', 'exorbitant', 'expiratie', 'expireren', 'explicatie', 'expliceren', 'expliciet', 'exploitatie', 'exploiteren', 'exploratie', 'exploreren', 'explosief', 'exponentieel', 'exposé', 'extensivering', 'extenso', 'in', 'extramuraal', 'extreem', 'extremis', 'in', 'extrinsiek', 'exuberant', 'faciliteren', 'fasegewijs', 'fiat', 'fiatteren', 'financiële middelen', 'finetunen', 'finetuning', 'flankerend', 'flankerend beleid', 'fluctueren', 'fortiori', 'a', 'fraai', 'fundament', 'fundamenteel', 'funderen', 'fungeren', 'gaarne', 'garant staan voor', 'geaccordeerd', 'geagendeerd', 'gecompliceerd', 'gecoördineerd', 'geenszins', 'gefaseerd', 'gegrond op', 'geïntegreerde aanpak', 'geïnvolveerd', 'gekant zijn tegen', 'gelegenheid', 'niet in de  zijn', 'gelet op', 'gelieve', 'gelijkluidend', 'gelijktijdig', 'gelukken', 'gemeentelijke verordening', 'gemeenzaam', 'gemotiveerd', 'genereren', 'genoegzaam', 'geraken', 'gerealiseerd', 'gereed', 'gerevitaliseerd', 'geschieden', 'geschil', 'gestructureerd', 'gewag maken van', 'gezien', 'gezien het feit dat', 'gremium', 'gremia', 'grootstedelijk', 'hangende het besluit', 'heden', 'hedenmiddag', 'hedenochtend', 'heden ten dage', 'heroriëntatie', 'heroriënteren', 'heroverwegen', 'hetgeen', 'het is geboden', 'het kan niet worden tegengesproken dat', 'het ligt geenszins in de bedoeling dat', 'het moge duidelijk zijn', 'het valt te proberen', 'het ware te wensen dat', 'hetwelk', 'hiërarchie', 'hiernavolgende', 'hieromtrent', 'hieronder ressorteert', 'hier te lande', 'hiertoe', 'hoewel', 'hogergenoemde', 'hoofde', 'uit dien', 'hoofdlijnen', 'hoofdzakelijk', 'hoogachtend', 'horizontaal werkverband', 'houdende', 'immers', 'impact', 'implementatie', 'implementeren', 'implicatie', 'impliceren', 'in aanmerking nemen', 'in beginsel', 'in casu', 'incident', 'in concreto', 'incongruent', 'inconsequent', 'incourant', 'in de betekenis van', 'in de buurt van', 'in de gelegenheid zijn', 'in de nabije toekomst', 'in de nabijheid van', 'in de omstandigheid verkeren', 'in de trant van', 'in dezen', 'in de week leggen', 'in de zin van', 'indicatie', 'indien', 'in dier voege', 'in duplo', 'inflatie', 'in gemeen overleg', 'ingeval', 'in geval van', 'ingevolge', 'in goede orde', 'in het huidige tijdsgewricht', 'in het kader van', 'in het licht van', 'in het merendeel van de gevallen', 'in het navolgende', 'inhoudelijk', 'initiatief', 'initieel', 'initiëren', 'in mindere mate', 'innovatie', 'in overleg treden', 'in overweging nemen', 'in samenwerking met', 'insisteren', 'in situ', 'institueren', 'institutionaliseren', 'integraal', 'integreren', 'intentie', 'in toenemende mate', 'in verband met', 'insisteren', 'invorderen', 'in werking stellen', 'in werking treden', 'inwilligen', 'inzake', 'irreëel', 'irrelevant', 'issue', 'item', 'jegens', 'jongstleden', 'jumelage', 'juncto', 'jurisprudentie', 'justificatie', 'kandelaberen', 'kennisnemen van', 'kerntaak', 'kickoff', 'klankborden', 'kortsluiten', 'kortwieken', 'kostenindicatie', 'krachtens', 'kwalificatie', 'kwalificeren', 'kwalijk', 'kwalitatief', 'kwantificeren', 'kwantitatief', 'kwantiteit', 'kwartaal', 'kwestie', 'kwestieus', 'laatstelijk', 'labelen', 'laken', 'lanceren', 'landen', 'langjarig', 'larderen met', 'Lectori Salutem', 'leges', 'legio', 'legitiem', 'legitimatie', 'legitimeren', 'leidmotief', 'leitmotiv', 'lering trekken uit', 'liaison', 'liberalisatie', 'liberalisering', 'licentie', 'lichaam', 'lichtvaardig', 'lijntjes leggen', 'litigieus', 'locatie', 'managen', 'mandaat', 'manifest worden', 'mede', 'mededelen', 'medio', 'meenemen', 'menigeen', 'merendeels', 'met als reden', 'met als resultaat dat', 'met behulp van', 'met betrekking tot', 'met de bedoeling dat', 'met het oog op', 'met het resultaat dat', 'met name', 'met redenen omkleed', 'met referte aan', 'met terzijde laten van', 'met weglating van', 'middelen', 'middels', 'Mijne heren', 'mijns inziens', 'mits', 'mitsdien', 'mitsgaders', 'moge', 'momenteel', 'monitoren', 'motie', 'motie van wantrouwen', 'motiveren', 'mutaties', 'naar aanleiding van', 'naar behoren', 'naar  toe', 'nadere', 'nadien', 'nagenoeg', 'navolgende', 'navrant', 'negental', 'neveneffect', 'nevenstaande', 'nevenvermelde', 'nietig', 'niettegenstaande', 'niettemin', 'niet zijnde', 'nimmer', 'nochtans', 'noodzakelijk', 'nopen', 'nopens', 'of', 'en zo ja', 'op welke wijze', 'officieus', 'ofschoon', 'oftewel', 'ombuigen', 'ombuiging', 'om deze redenen', 'om niet', 'om reden van', 'om te komen tot', 'omtrent', 'omvangrijk', 'onbezoldigd', 'ondanks het feit dat', 'onder curatele stellen', 'ondergetekende', 'onderhavig', 'onder invloed van', 'onder referte aan', 'onderscheidenlijk', 'onderuitputting', 'onder verwijzing naar', 'onderwerpelijk', 'onderwijl', 'ongeacht', 'ongenoegen', 'onjuist', 'onlangs', 'ontberen', 'onthouden', 'ontplooien', 'ontstentenis', 'bij  van', 'onverlet', 'dat laat', 'onverwijld', 'onzerzijds', 'onzes inziens', 'oogmerk', 'oorzaak ligt in het feit dat', 'op basis van', 'op deze wijze', 'op een prettige', 'goede', 'slechte', 'wijze', 'op grond van', 'op grond van het voorgaande', 'op het gebied van', 'opportuniteiten', 'opportuun', 'op het huidige moment', 'opstarten', 'opteren voor', 'optimaal', 'optimaliseren', 'op voorhand', 'op welke wijze', 'over de gehele linie', 'overeenkomstig', 'overhead', 'overigens', 'panel', 'paraaf', 'paraferen', 'participeren', 'partieel', 'peildatum', 'per abuis', 'percent', 'per omgaande', 'persisteren', 'piketpaaltjes slaan', 'pilot', 'pogen', 'portefeuille', 'prealabel', 'precedent', 'precedentwerking', 'preliminair', 'preluderen op', 'prematuur', 'prevaleren', 'preventief', 'preventieve maatregelen', 'primair', 'principe', 'principieel', 'prioriteit', 'prioriteitsstelling', 'prioriteren', 'proactief', 'procedé', 'procedure', 'procedureel', 'profileren', 'prognose', 'prominent', 'qua', 'quasi', 'quickscan', 'quod non', 'quotum', 'ramen', 'raming', 'randvoorwaarden', 'ratio', 'rationeel', 'reactief', 'realiseren', 'recentelijk', 'reces', 'rechtens', 'reclamant', 'rectificeren', 'reden daartoe is', 'reductie', 'reeds', 'referentie', 'refereren aan', 'reflecteren', 'regarderen', 'regie nemen', 'regulier', 'relationeel', 'relevant', 'reparatiewetgeving', 'repercussie', 'repliceren', 'respectievelijk', 'respons', 'restitutie', 'restricties', 'resulteren in', 'resumé', 'resumerend', 'retourneren', 'retourzenden', 'retributie', 'revitalisering', 'ruchtbaarheid geven aan', 'sancties', 'sanctioneren', 'schoon', 'schrijven', 'sedert', 'separaat', 'seponeren', 'shared services', 'significant', 'slechts', 'specifiek', 'speerpunt', 'spinoff', 'stagnatie', 'stagneren', 'stakeholder', 'stellen', 'storneren', 'stornering', 'strategie', 'stringent', 'structureel', 'stukje', 'een  beleid', 'vooruitgang', 'subjectief', 'suboptimaal', 'subsidiabel', 'substantieel', 'summier', 'suppletoir', 'tal van', 'target', 'taskforce', 'te', 'te allen tijde', 'technisch', 'te dezen', 'tegemoetzien', 'tegen de achtergrond van', 'telkenmale', 'temporiseren', 'ten aanzien van', 'ten behoeve van', 'ten bewijze van', 'ten dele', 'ten detrimente van', 'teneinde', 'ten gevolge van', 'ten grondslag liggen aan', 'ten gunste van', 'ten minste', 'ten principale', 'ten tijde van', 'ten uitvoer brengen', 'ten volle van bewust', 'zich er  zijn', 'tenzij', 'te onzent', 'ter bereiking hiervan', 'ter beschikking hebben', 'terdege', 'terechtwijzen', 'terechtwijzing', 'ter gelegenheid van', 'ter hand nemen', 'ter hand stellen', 'ter realisering van dit oogmerk', 'terstond', 'terugkoppelen', 'terugmelden', 'ter zake', 'ter zake kundig', 'ter zake van', 'ter zitting', 'te uwen name', 'te uwent', 'tevens', 'tezamen', 'te zijner tijd', 'thans', 'tiental', 'tijdpad', 'tijdsbestek', 'toekomen', 'doen', 'toentertijd', 'toetsen aan', 'topic', 'tot de conclusie komen', 'tot taak hebben', 'trachten', 'transparant', 'tweetal', 'u gelieve', 'uit anderen hoofde', 'uit het oogpunt van', 'uit hoofde van', 'ultimo', 'universeel', 'urgent', 'urgentie', 'usance', 'utilitair', 'uwenthalve', 'uwentwege', 'uwerzijds', 'vacant', 'vacaturehoudend', 'valide', 'valideren', 'valorisatie', 'vandaar', 'van de zijde van', 'van gemeentewege', 'van mening zijn', 'van oordeel zijn', 'van plan zijn', 'van start gaan', 'van tevoren', 'vanuit de organisatie', 'vanwege', 'vanwege het feit dat', 'veelal', 'veelszins', 'veelvuldig', 'verbeterpunten', 'vergewissen', 'zich ervan', 'verketelen', 'verkiezen boven', 'vermanen', 'vermogen', 'verordening', 'verstenen', 'verstreken periode', 'vertaalslag', 'vertrouwen opzeggen', 'verwerven', 'verzuimen', 'viertal', 'vigeren', 'vigerend', 'vijftal', 'vlieguren maken', 'voldoen', 'volgaarne', 'volgtijdelijk', 'voorafgaand aan', 'voorafgaandelijk', 'vooraleer', 'vooralsnog', 'voorhanden zijn', 'voorheen', 'voor het geval dat', 'voorliggend', 'voormeld', 'voornemen', 'voornemens zijn', 'voornoemde', 'voornoemde werkzaamheden', 'voorshands', 'voorts', 'voorwaarden scheppen', 'voor wat betreft', 'vorderen', 'vorenbedoelde', 'vorenomschreven', 'vorenoverwogene', 'vorenstaande', 'vraagpunt', 'vrezen voor', 'vrijwel', 'waarborgen', 'waarnemen', 'weder', 'wederom', 'wat betreft', 'watergang', 'weliswaar', 'welke', 'wellicht', 'wensen', 'werkbaar', 'werkvoorraad', 'werkzaam zijn', 'weshalve', 'wij vertrouwen erop u hiermee voldoende te hebben geïnformeerd', 'wijze', 'willens en wetens', 'woonachtig zijn', 'woonvoorziening', 'wortelopdruk', 'xenofobie', 'xenofoob', 'yell', 'yup', 'zelfredzaamheid', 'zestal', 'zevental', 'zich beraden', 'zich verstaan met', 'zien op', 'zij', 'hoe dit ook', 'zij', 'wat daarvan', 'zijde', 'van de  van', 'zijdens', 'zodanig', 'zodoende', 'zonder meer', 'zonder uitzondering', 'zorg dragen', 'zorgvuldig', 'zo spoedig mogelijk', 'zulks', 'zulks impliceert derhalve', 'zwaktebod']

informal_list_final = ['goed', 'met', 'door', 'over', 'omdat', 'groot', 'belangrijk', 'vrij sterk', 'aanpassen', 'veranderen', 'dadelijk', 'gauw', 'binnenkort', 'leiding geven', 'sturen', 'te sluiten', 'mensen in dienst nemen', 'begin', 'start', 'beginnen', 'van start gaan', 'starten', 'beginnen', 'starten', 'eerst', 'eerder', 'in het begin', 'benaderen', 'aanpakken', 'gebruiken', 'per ongeluk', 'door een vergissing', 'door een fout', 'op andere dingen letten', 'stoppen met', 'nadruk leggen op iets anders', 'goedkeuring', 'gebouw', 'locatie', 'instemmen met', 'vinden', 'van mening zijn', 'acht', 'doorgaan met', 'blijven werken aan', 'aanpassen', 'moderniseren', 'direct', 'onmiddellijk', 'van', 'toegevoegd', 'aanvullend', 'extra', 'passend', 'juist', 'goed', 'instemming', 'steun', 'instemmen', 'steunen', 'laten merken dat je het ermee eens bent', 'direct', 'plaatselijk', 'tijdelijk', 'verduidelijken', 'toelichten', 'bewijzen', 'adviseren', 'de raad geven om', 'afhandeling', 'afsluiting', 'bevestigend', 'behalve', 'niet meegerekend', 'als je niet let op', 'zonder  mee te tellen', 'afmaken', 'beëindigen', 'controleren', 'beëindigen', 'verminderen', 'daar', 'op die plaats', 'algauw', 'binnenkort', 'zo snel als', 'of', 'wel of niet', 'zo', 'op die manier', 'hoewel', 'maar', 'in alle opzichten', 'zeker', 'hoe dan ook', 'toewijzen', 'overal', 'dan', 'in dat geval', 'op dat moment', 'en', 'net als', 'net als', 'en', 'of', 'destijds', 'in die tijd', 'toen', 'voordat', 'waar', 'dubbel', 'twijfel', 'op een andere manier', 'anders', 'aan de andere kant', 'belangstelling', 'aankondiging', 'verwachten', 'vooruitlopen op', 'partij die in hoger beroep gaat', 'iemand die laat weten het er niet mee eens te zijn', 'in hoger beroep gaan', 'protesteren', 'een beroep doen op', 'aanspreken', 'waardering', 'vooraf', 'van tevoren', 'zelfstandig', 'onafhankelijk', 'nogal', 'aanzienlijk', 'in de smaak vallen', 'prettig vinden', 'gaan over', 'bevatten', 'inhouden', 'is nodig', 'is gevraagd', 'moeten', 'hoeven', 'nodig hebben', 'nodig zijn', 'willen', 'moeten', 'horen', 'behalve', 'met uitzondering van', 'met behoud van', 'onder voorbehoud van', 'tenzij', 'behalve als', 'tenzij', 'behalve als', 'tenzij', 'behalve als', 'krijgen', 'betrokken persoon', 'betrokkene', 'goed plan', 'extra aandacht geven aan', 'meer geld geven aan', 'prestaties vergelijken', 'behalve', 'naast', 'en ook', 'net als', 'evenals', 'met daarbij', 'bedoelen', 'als doel hebben', 'proberen te bereiken', 'een', 'een zeker', 'dit', 'dat', 'wat hier staat', 'laten weten', "op z'n kop geven", 'zeggen dat iemand iets verkeerd heeft aangepakt', 'spijt', 'papieren', 'documenten', 'gebeuren', 'uitvoeren', 'voortzetten', 'vasthouden', 'gaan over', 'duurt van  tot', 'over', 'over', 'onderwerp', 'gaan over', 'verklaren', 'zeggen dat', 'zorgen voor', 'doen', 'bekijken', 'doen', 'aan iets werken', 'gebruiken', 'betaald', 'schriftelijk', 'in de', 'een brief', 'met de brief van', 'hierbij', 'meewerken', 'betalen', 'bij', 'voor', 'af en toe', 'soms', 'soms', 'toevallig', 'als', 'dus', 'daarom', 'door', 'bij afwezigheid van', 'als x er niet is', 'als x niet het geval is', 'door', 'in een brief van 1 maart', 'als', 'op die manier', 'van twee kanten', 'bij de gemeente', 'binnen onze werkwijze', 'wij eisen', 'onveilige verkeersplek', 'gezien', 'zoals blijkt uit', 'vasthouden aan', 'zorgen dat iets  blijft bestaan', 'voortgezet wordt', 'deze', 'dit', 'deze', 'dit', 'wat hierboven staat', 'reclame maken voor', 'als een merk', 'vaste  eigenschap presenteren', 'tankstation', "meestal te vervangen door'en'of'of'", 'vermogen', "meestal te vervangen door'en'of'of'", 'soort', 'rubriek', 'commissie', 'ongeveer', 'stadspromotie', 'bepaling', 'regel', 'voorbehoud', 'samenvoegen', 'samenhang', 'binding', 'uitwisselbaar', 'verenigbaar', 'vergoeding', 'schade goedmaken', 'moeilijk', 'ingewikkeld', 'gebouw', 'ingewikkeld maken', 'moeilijk maken', 'ontwerp', 'schets', 'opzet', 'plan', 'ontwerpen', 'schetsen', 'zwanger worden', 'praktisch', 'duidelijk', 'voorwaarde', 'samenstelling', 'botsen', 'strijden', 'ruzie', 'conflict', 'in overeenstemming met', 'zoals', 'volgens', 'overeenstemmend', 'overeenstemming', 'zoals je je had voorgenomen', 'zoals was afgesproken', 'onverstoorbaar en rechtdoorzee', 'logisch', 'samenhangend', 'het vasthouden', 'het vastleggen van iets goeds', 'het bewaren van', 'zien', 'vaststellen', 'samenstelling', 'stand van zaken', 'instellen', 'vormen', 'opbouwend', 'bruikbaar', 'nuttig', 'advies vragend', 'advies vragen', 'hulp zoeken', 'doorgaan', 'volhouden', 'overeenkomst', 'afspraak met', 'onderdelen op elkaar afstemmen', 'de samenwerking regelen', 'gangbaar', 'gebruikelijk', 'maken', 'norm', 'richtlijn', 'eis', 'noodzakelijk', 'belangrijk', 'met anderen', 'met partners', 'toenemend', 'oplopend', 'vreemd', 'merkwaardig', 'aangezien', 'omdat', 'bovendien', 'behalve dat', 'daarbij', 'verder', 'datum', 'of', 'nogal', 'zeer', 'bedanken', 'dus', 'niet centraal', 'niet vanuit één kantoor', 'gebouw', 'van', 'afgeleid', 'afleidend', 'tot de conclusie komend', 'in feite', 'degene', 'samenwerken', 'een ander laten doen', 'een taak neerleggen bij', 'aan een ander geven', 'vinden', 'denken', 'van de', 'regels afschaffen', 'versimpelen', 'daarom', 'om die reden', 'dan ook', 'dus', 'vertraging', 'uitstel', 'mislopen', 'verliezen', 'verlies', 'toch', 'maar', 'over', 'de  waarom het gaat', 'dit', 'wensen', 'wens', 'toen we daarnaar vroegen', 'echter', 'maar', 'toch', 'hoewel', 'toch', 'toen', 'daarom', 'de afgelopen tijd', 'deze week', 'maand', 'nu', 'binnenkort', 'van mijn', 'onze kant', 'ik heb', 'wij hebben', 'beslissingen', 'beslissing', 'hierover', 'daarover', 'wat dat betreft', 'moeten', 'zo', 'daardoor', 'onderscheiden', 'verschillen', 'verschil maken', 'onderbroken', 'niet aan één stuk door', 'afkeuring', 'verschil', 'tegenstrijdigheid', 'afwijking', 'iets wat niet logisch is', 'verspreiden', 'versturen', 'verdelen', 'deze brief', 'verschillend', 'ongelijkheid', 'allerlei', 'maar', 'toesturen', 'opsturen', 'sturen', 'streng', 'rechtlijnig', 'door', 'met', 'doorgaan', 'plaatsvinden', 'verder ontwikkelen', 'doorgaan', 'volhouden', 'niet stoppen', 'steun', 'treuzelen', 'afwachten', 'drie', 'zo', 'maar', 'toch', 'maar', 'toch', 'leerzaam', 'enkele', 'of: precies aantal noemen', 'binnenkort', 'deze', 'dit alles', 'enkele', 'of: precies getal noemen', 'deze', 'dit', 'pas dan', 'uitvoeren', 'doen', 'ergens anders', 'uitschakelen', 'verwijderen', 'iedereen', 'aan de ene kant', 'als je het zo bekijkt', 'een paar', 'enkele', 'een beetje', 'wat', 'alleen', 'beginnen', 'in gang zetten', 'brief', 'gelijkwaardig', 'gelijk', 'dus', 'ertoe dienen', 'erger worden', 'hogerop zoeken', 'hogerop brengen', 'onmisbaar', 'nodig', 'noodzakelijk', 'bespreking', 'beoordeling', 'bespreken', 'beoordelen', 'achteraf bespreken om ervan te leren', 'ook niet', 'maar', 'toch', 'ook', 'net zo', 'ook', 'net zo', 'op dezelfde manier', 'duidelijk', 'meteen duidelijk', 'volgens', 'op grond van', 'uitzonderlijk', 'een samenvatting maken', 'overdreven', 'veel te veel', 'hoog', 'meteen', 'van nu af aan', 'direct', 'buitensporig', 'afloop', 'beëindiging', 'einde', 'verlopen', 'vervallen', 'aflopen', 'eindigen', 'uitleg', 'verklaring', 'verklaren', 'uitleggen', 'duidelijk maken', 'uitdrukkelijk', 'nadrukkelijk', 'benutting', 'misbruik', 'winstgevend maken', 'benutten om een bedrijf te voeren', 'misbruik maken van', 'het onderzoeken', 'het doorzoeken', 'onderzoeken', 'opsporen', 'plotseling enorm groeiend', 'toenemend', 'enorm sterk', 'enorm krachtig', 'overzicht', 'toelichting', 'presentatie', 'uitbreiding', 'in zijn geheel', 'helemaal', 'buiten de muren van', 'in de wijk', 'bij de mensen thuis', 'buitengewoon', 'zeer ongewoon', 'in heel hoge mate', 'op het allerlaatste moment', 'niet tot de kern behorend', 'aan de buitenkant', 'uiterlijk', 'overdadig', 'overdreven', 'ondersteunen', 'mogelijk maken', 'in delen', 'in stappen', 'toestemming', 'goedkeuring', 'goedkeuren', 'bekrachtigen', 'geld', 'verbeteren', 'regeling van details', 'precieze afstemming', 'daarnaast', 'extra', 'bijbehorend', 'beleid dat helpt om een maatregel', 'wet goed te kunnen uitvoeren', 'telkens op en neergaan', 'schommelen', 'sterker nog', 'met nog meer reden', 'al helemaal', 'mooi', 'goed', 'basis', 'grondslag', 'de basis betreffend', 'diepgaand', 'baseren', 'dienstdoen als', 'optreden als', 'graag', 'beloven', 'goedgekeurd', 'op de agenda gezet', 'als agendapunt opgenomen', 'ingewikkeld', 'afgestemd', 'in goede samenwerking', 'niet', 'zeker niet', 'helemaal niet', 'in delen', 'in stappen', 'gebaseerd op', 'aanpak als één geheel', 'algemene aanpak', 'betrokken', 'tegen iets zijn', 'bezwaren hebben tegen', 'niet kunnen', 'met het oog op', 'omdat', 'doordat', 'wilt u', 'gelijk', 'hetzelfde', 'tegelijk', 'lukken', 'regels van de gemeente', 'informeel', 'alledaags', 'waarbij redenen worden', 'zijn gegeven', 'met een goede uitleg', 'maken', 'ontwikkelen', 'ervoor zorgen dat iets er komt', 'voldoende', 'algemeen', 'krachtig', 'raken', 'gemaakt', 'gebouwd', 'besloten', 'afgesproken', 'enz', 'klaar', 'af', 'opgeknapt', 'weer tot leven gebracht', 'gebeuren<br', '>', 'ruzie', 'conflict', 'systematisch opgebouwd', 'met een goede opbouw', 'in fasen verdeeld', 'melden', 'bekendmaken', 'daarom', 'met het oog op', 'omdat', 'overleg', 'vergadering', 'groep', 'vergadering van beleidsmedewerkers', 'in', 'van grote steden', 'zolang nog niets is besloten', 'nu', 'vandaag', 'vanmiddag', 'eerder', 'later op deze middag', 'vanochtend', 'eerder', 'later op deze ochtend', 'vandaag', 'vandaag de dag', 'tegenwoordig', 'opnieuw bekijken', 'opnieuw de juiste richting bepalen', 'nog een keer over iets nadenken', 'wat', 'dat wat', 'het moet', 'het is nodig', 'het is zo dat', 'het is waar dat', 'hoeft niet', 'ik bedoel niet', 'het is duidelijk', 'u weet', 'we kunnen het proberen', 'u kunt het proberen', 'het is mogelijk', 'we hopen', 'het zou beter zijn als', 'dat', 'dat wat', 'rangorde', 'volgende', 'hierover', 'hieronder valt', 'in ons land', 'hiervoor', 'om dit te doen', 'met dit doel', 'maar toch', 'hiervoor genoemde', 'om die reden', 'daarom', 'wat hieruit voortvloeit', 'de belangrijkste punten', 'vooral', 'met vriendelijke groet', 'samenwerken op één niveau', 'met', 'toch', 'ook', 'namelijk', 'invloed', 'effect', 'uitvoering', 'invoering', 'toepassing', 'invoeren', 'uitvoeren', 'ervoor zorgen dat iets wordt toegepast', 'gevolg', 'inhouden', 'betekenen', 'rekening houden met', 'in de eerste plaats', 'eigenlijk', 'liever', 'in dit geval', 'in uw geval', 'in het geval dat', 'gebeurtenis', 'vervelende gebeurtenis', 'dus', 'in werkelijkheid', 'in een bepaald geval', 'met duidelijke voorbeelden', 'verschillend', 'ongelijk', 'afwijkend', 'niet volgens afspraak', 'ongewoon', 'zo', 'zoals', 'ongeveer', 'kunnen', 'binnenkort', 'snel', 'vlakbij', 'in de buurt van', 'zijn', 'hebben', 'zo', 'zoals', 'hierin', 'wat dit betreft', 'voorbereiden', 'alvast beginnen', 'voorstellen', 'zo', 'zoals', 'aanwijzing', 'als', 'wanneer', 'zo', 'op die manier', 'in tweevoud', 'waardevermindering van geld', 'prijsstijging', 'samen', 'in overleg met', 'als', 'wanneer', 'bij', 'als', 'door', 'op grond van', 'naar aanleiding van', 'als gevolg van', 'goed', 'nu', 'in de tijd waarin we nu leven', 'binnen', 'om', 'op basis', 'daarom', 'meestal', 'bijna altijd', 'hierna', 'hieronder', 'over de inhoud', 'als het om de inhoud gaat', 'plan', 'idee', 'eerste stap', 'eerst', 'om te beginnen', 'eigenlijk', 'beginnen', 'in gang zetten', 'minder', 'vernieuwing', 'overleggen', 'nadenken over', 'met', 'aandringen op', 'ter plekke', 'meteen', 'inrichten', 'vastleggen', 'invoeren', 'volledig', 'helemaal', 'inpassen', 'aanpassen', 'bij elkaar brengen', 'bedoeling', 'we willen', 'we proberen', 'steeds vaker', 'door', 'omdat', 'aandringen op', 'betaling eisen', 'beginnen', 'starten', 'geldt vanaf', 'beginnen', 'toestaan', 'toestemming geven', 'toestemming verlenen aan']

informal_list_extended = ['zich schikken in of naar', 'toegeven', 'over', 'onwerkelijk', 'niet realistisch', 'niet ter zake doend', 'zonder betekenis', 'niet van toepassing', 'belangrijk onderwerp', 'het punt waar het om gaat', 'onderwerp', 'punt', 'tegenover', 'onlangs', 'vriendschapsband tussen steden', 'dorpen', 'in verband met', 'wat blijkt uit eerdere beslissingen van een rechtbank', 'verdediging', 'rechtvaardiging', 'ingrijpend snoeien', 'horen', 'lezen', 'belangrijkste taak', 'aftrap', 'opening', 'feestelijke start', 'overleggen', 'praten', 'snel afspreken', 'zorgen dat iedereen weet wat er aan de hand is', 'wat er moet gebeuren', 'kleiner maken', 'beperken', 'berekening van de kosten', 'op grond van', 'volgens de wet', 'beoordeling', 'een naam geven aan', 'het recht hebben iets te doen', 'ergens aan mee te doen', 'niet goed', 'slecht', 'wat de kwaliteit', 'waarde betreft', 'bepalen', 'meten', 'in cijfers vaststellen of uitdrukken', 'als het gaat om de hoeveelheid', 'grootte', 'hoeveelheid', 'bedrag', 'grootte', 'een vierde deel van het jaar', 'periode van drie maanden', 'de zaak waar het om gaat', 'probleem', 'twijfelachtig', 'onlangs', 'op', 'noemen', 'sterk afkeuren', 'naar voren brengen', 'onder de aandacht brengen', 'begrepen worden', 'geaccepteerd worden', 'jarenlang', 'voorzien van', 'Geachte', 'Beste', 'geld', 'kosten', 'heel veel', 'ontelbaar', 'wettelijk', 'wettig', 'rechtmatig', 'terecht', 'goed', 'verklaring dat iets wettig', 'echt is', 'bewijs dat iemand echt degene isdie hij zegt dat hij is', 'wettigen', 'verantwoorden', 'aantonen dat iets goed is', 'bewijzen dat iemand echt degene is die hij zegt dat hij is', 'grondgedachte', 'basis', 'uitgangspunt', 'wijzer worden van', 'iets als een wijze les zien', 'gebruiken', 'verbintenis', 'het meer ruimte geven', 'verruiming', 'vergunning', 'vergunningsbewijs', 'organisatie', 'college', 'overleg', 'ondoordacht', 'zonder goed te hebben nagedacht', 'overlegd', 'verbinden met', 'twijfelachtig', 'plaats', 'gebouw', 'besturen', 'doen', 'uitvoeren', 'regelen', 'toestemming', 'duidelijk worden', 'ook', 'onder andere', 'laten weten', 'vertellen', 'schrijven', 'halverwege de maand', 'half', 'onthouden', 'erbij betrekken', 'veel mensen', 'velen', 'de meeste', 'de meesten', 'vooral', 'omdat', 'zodat', 'met', 'door', 'over', 'om', 'daarom', 'om', 'daarom', 'zodat', 'vooral', 'met een goede uitleg', 'waarbij precies is', 'wordt verteld wat de reden is', 'verwijzend naar', 'zonder', 'zonder', 'geld', 'bedrag', 'door', 'Geachte heer', 'mevrouw', 'Dames en heren', 'volgens mij', 'ik vind', 'ik denk', 'als', 'op voorwaarde dat', 'daarom', 'daardoor', 'bovendien', 'en', 'evenals', 'behalve', 'met', 'naast', 'net als', 'verder', 'mag', 'kan', 'nu', 'bekijken', 'bestuderen', 'in de gaten houden', 'politiek voorstel', 'vragen om het vertrek van', 'redenen geven', 'uitleggen waarom', 'wijzigingen', 'omdat x is gebeurd', 'na', 'goed genoeg', 'voor', 'verdere', 'uitgebreidere', 'daarna', 'vervolgens', 'later', 'bijna', 'volgende', 'hierna beschreven', 'pijnlijk', 'negen', 'onverwacht effect', 'onverwacht gevolg', 'hiernaast', 'hiernaast staande', 'onderstaande', 'hiernaast genoemde', 'ongeldig', 'onbelangrijk', 'desondanks', 'hoezeer', 'hoewel', 'ondanks', 'toch', 'intussen', 'desondanks', 'maar geen', 'nooit', 'echter', 'toch', 'nodig', 'dwingen', 'noodzaken', 'noodzakelijk maken', 'over', 'onderwerp', 'of', 'en zo ja', 'hoe', 'of het zo is', 'en áls het zo is', 'hoe', 'niet officieel', 'hoewel', 'al', 'toch', 'maar', 'of', 'bezuinigen', 'bezuiniging', 'daarom', 'gratis', 'omdat', 'om', 'over', 'groot', 'niet betaald', 'hoewel', 'toch', 'laten controleren', 'iemand anders aanwijzen om beslissingen te nemen voor iemand', 'ik', 'bij mij', 'deze', 'die', 'dit', 'dat', 'door', 'zoals in  staat', 'duidelijk', 'er is geld over', 'x maakt', 'maken te weinig gebruik van', 'over', 'deze', 'dit', 'intussen', 'of nu wel of niet', 'klacht', 'boosheid', 'ontevredenheid', 'verkeerd', 'fout', 'kortgeleden', 'op', 'eraan ontbreken', 'niet hebben', 'niet geven', 'niet toekennen', 'ontwikkelen', 'uitvoeren', 'bij afwezigheid van', 'als x er niet is', 'als x niet het geval is', 'dat betekent niet', 'dat verandert niets aan', 'onmiddellijk', 'direct', 'van ons', 'volgens ons', 'volgens ons', 'wij vinden', 'wij denken', 'doel', 'dit komt door', 'daarom', 'zo', 'zo', 'prettig', 'goed', 'slecht', 'vanwege', 'daarom', 'over', 'voor', 'kansen', 'goed om op dit moment te doen', 'op het juiste moment', 'nu', 'beginnen', 'kiezen voor', 'de voorkeur geven aan', 'zo goed mogelijk', 'verbeteren', 'eerst', 'vooraf', 'hoe', 'helemaal', 'zo', 'als', 'op grond van', 'volgens', 'bijkomende kosten', 'verder', 'daarnaast', 'voor de rest', 'onderzoeksgroep', 'handtekening', 'handtekening zetten', 'meedoen', 'deelnemen', 'gedeeltelijk', 'vanaf', 'sinds', 'per ongeluk', 'procent', 'meteen', 'direct', 'volhouden', 'vasthouden aan', 'grenzen aangeven', 'meteen duidelijk maken hoever men wil gaan', 'proefproject', 'proberen', 'taken', 'takenpakket', 'beleidsterrein', 'voorafgaand', 'voordat andere punten aan de orde komen', 'voorbeeld', 'eerder besluit', 'een mogelijk verkeerd voorbeeld gevend', 'voorafgaand', 'inleidend', 'alvast iets zeggen over iets wat later aan de orde komt', 'te vroeg', 'voor zijn beurt', 'voor laten gaan', 'om  te voorkomen', 'dingen die gedaan worden om iets te voorkomen', 'in de eerste plaats', 'belangrijkste', 'eerste', 'keuze', 'uitgangspunt', 'volgens een basisidee', 'volgens iemands overtuiging', 'uit de grond van iemands hart', 'voorrang', 'iets wat voor moet gaan', 'volgorde van belangrijkheid', 'iets voorrang geven', 'je eerst op x richten', 'actief', 'problemen vóór willen zijn', 'werkwijze', 'manier van doen', 'werkwijze', 'manier om iets te aan te pakken', 'wat de werkwijze betreft', 'volgens de afgesproken manier van werken', 'kenbaar maken', 'onderscheiden', 'voorspelling', 'belangrijk', 'over', 'rond', 'als het gaat om', 'schijnbaar', 'net alsof', 'snel onderzoek op hoofdlijnen', 'nattevingerwerk', 'maar dat is niet zo', 'dat is niet waar', 'aandeel', 'schatten', 'begroten', 'schatting', 'begroting', 'eisen', 'voorwaarden', 'verstand', 'oorzaak', 'rede', 'verhouding tussen', 'verstandig', 'redelijk', 'afwachten', 'te laat', 'achter de feiten aan', 'maken', 'bouwen', 'behalen', 'bereiken', 'onlangs', 'kortgeleden', 'net', 'vakantieperiode', 'volgens het recht', 'rechtmatig', 'degene die bezwaar maakt', 'de klager', 'de eiser', 'verbeteren', 'rechtzetten', 'dit wordt gedaan omdat', 'vermindering', 'korting', 'al', 'verwijzing', 'verwijzen naar', 'ergens op reageren', 'nadenken over', 'aangaan', 'leiden', 'gebruikelijk', 'gewoon', 'in relatie tot', 'binnen', 'belangrijk', 'wetgeving die ervoor moet zorgen dat een bestaande wet beter werkt', 'maatregel die ertegenin gaat', 'reactie', 'antwoorden', 'en', 'of', 'antwoord', 'reactie', 'teruggave', 'terugbetalen', 'u krijgt  euro terug', 'beperkingen', 'grenzen', 'leiden tot', 'samenvatting', 'samenvattend', 'terugsturen', 'terugsturen', 'bijdrage', 'het opknappen', 'nieuw leven inblazen', 'aandacht geven', 'bekendmaken', 'maatregelen', 'straf', 'boete', 'goedkeuren', 'zeggen dat iets moet doorgaan', 'óf: straffen', 'bestraffen', 'hoewel', 'al', 'de brief', 'sinds', 'vanaf', 'apart', 'los', 'niet vervolgen', 'er niets mee doen', 'samenwerking', 'centralisatie', 'deeltaken die op verschillende afdelingen worden uitgevoerd', 'allemaal laten uitvoeren door één overkoepelende afdeling', 'opvallend', 'statistisch verantwoord', 'alleen', 'vooral', 'in het bijzonder', 'belangrijkste onderdeel', 'punt waar we de meeste aandacht aan geven', 'gevolg', 'bijproduct', 'vertraging', 'stilstand', 'vertraagd raken', 'tot stilstand komen', 'betrokkene', 'betrokken partij', 'verklaren', 'meedelen', 'terugstorten', 'terugboeking', 'plan', 'aanpak', 'strak', 'streng', 'jaarlijks', 'belangrijk', 'voldoende', 'beleid', 'vooruitgang', 'persoonlijk', 'eigen mening', 'alleen maar op het eigen gevoel lettend', 'niet zo goed', 'minder goed dan verwacht', 'wat subsidie kan krijgen', 'wat geld van de overheid kan krijgen', 'belangrijk', 'flink', 'kort', 'beperkt', 'aanvullend', 'veel', 'doel', 'werkgroep', 'projectgroep', 'in', 'altijd', 'hier', 'hierin', 'hierbij', 'in dit geval', 'in dit opzicht', 'verwachten', 'krijgen', 'omdat', 'vanwege', 'telkens', 'uitstellen', 'vertragen', 'over', 'om', 'voor', 'bestemd voor', 'gericht aan', 'als bewijs van', 'om  te bewijzen', 'gedeeltelijk', 'op kosten van', 'om', 'als gevolg van', 'door', 'de reden is', 'voor', 'minstens', 'in elk geval', 'minimaal', 'wat de kern van de zaak betreft', 'wat de hoofdzaak betreft', 'toen', 'op het moment van', 'in de tijd van', 'uitvoeren', 'laten uitvoeren', 'goed weten', 'goed begrijpen', 'maar niet als', 'bij ons', 'om dit te bereiken', 'hebben', 'bezitten', 'uiteraard', 'wel', 'nauwkeurig', 'iemand op de vingers tikken', 'iemand zeggen dat hij een fout maakt', 'tik op de vingers', 'bij', 'omdat', 'beginnen', 'geven', 'om dit te bereiken', 'meteen', 'direct', 'onmiddellijk', 'zo meteen', 'straks', 'informeren', 'doorgeven wat besloten', 'besproken is', 'melden', 'over', 'deskundig', 'goed in zijn', 'haar vak', 'in', 'als het gaat om', 'wat  betreft', 'op', 'tijdens de zitting', 'op uw naam', 'bij u', 'bovendien', 'ook', 'zowel  als', 'samen', 'op hetzelfde ogenblik', 'samen', 'ooit', 'in de toekomst', 'later', 'nu', 'op dit moment', 'tien', 'planning', 'tijd', 'periode', 'sturen', 'toen', 'vergelijken met', 'onderwerp', 'na  nadenken een beslissing nemen', 'de conclusie trekken dat', 'moeten', 'proberen', 'open', 'doorzichtig', 'te controleren', 'twee', 'wij vragen u', 'om een andere reden', 'om', 'door', 'op grond van', 'namens', 'eind', 'op de laatste dag van de maand', 'algemeen', 'dringend', 'iets wat haast heeft', 'haast', 'het spoedeisend zijn', 'gebruik', 'gericht op praktisch nut', 'voor u', 'wat u betreft', 'van uw kant', 'in uw naam', 'van u', 'door u', 'van uw kant', 'vrij', 'waar een vacature bestaat', 'geldig', 'krachtig', 'verantwoord', 'betrouwbaar', 'goedkeuren', 'kennis echt gaan benutten', 'kennis toepassen in de praktijk', 'daarom', 'door', 'van', 'door de gemeente', 'door ons', 'vinden', 'vinden', 'willen', 'beginnen', 'vooraf', 'omdat', 'omdat', 'vaak', 'in veel opzichten', 'op veel gebieden', 'vaak', 'dingen die beter moeten', 'fouten', 'zekerheid moeten hebben over', 'zeker moeten weten dat', 'ervoor zorgen iets zeker te weten', 'uitzoeken', 'een cvketel vervangen', 'kiezen voor iets anders', 'iets anders een betere keuze vinden', 'op de vingers tikken', 'de les lezen', 'kunnen', 'in staat zijn', 'bezit', 'regels waaraan men zich moet houden', 'bindende regeling', 'bestraten', 'afgelopen tijd', 'uitwerking voor', 'wegsturen', 'ontslaan', 'kopen', 'krijgen', 'niet doen wat je zou moeten doen', 'je niet aan de afspraak houden', 'vier', 'gelden', 'geldend', 'huidig', 'vijf', 'werken', 'aan de gang zijn', 'ervaring opdoen', 'betalen', 'graag', 'na elkaar', 'eerder', 'vooraf', 'vooraf', 'voordat', 'voorlopig', 'tot nu toe', 'beschikbaar zijn', 'vroeger', 'eerder', 'als', 'deze', 'dit', 'hierboven vermeld', 'plan', 'van plan zijn', 'willen', 'deze', 'dit', 'eerdergenoemde', 'de eerdergenoemde werkzaamheden', 'deze werkzaamheden', 'voorlopig', 'nu', 'verder', 'bovendien', 'en', 'ook', 'verder', 'mogelijk maken', 'over', 'eisen', 'deze', 'dit', 'deze', 'dit', 'wat hiervoor uiteen is gezet', 'de overwegingen die hiervoor vermeld zijn', 'staan', 'hiervoor genoemde', 'deze', 'dit', 'voorgaande', 'vorige', 'vraag', 'bang zijn voor', 'bijna', 'zorgen dat iets blijft bestaan', 'vasthouden', 'zien', 'tijdelijk de functie hebben van', 'weer', 'weer', 'opnieuw', 'over', 'sloot', 'gracht', 'natuurlijk', 'die', 'dat', 'misschien', 'willen', 'praktisch', 'achterstand', 'werken', 'daarom', 'heeft u nog vragen', 'bel dan gerust met', 'manier', 'bewust', 'toch', 'wonen', 'woning', 'wortels van bomen die de stoep', 'het wegdek omhoogduwen', 'angst voor buitenlanders', 'hekel aan buitenlanders', 'vreemdelingenhaat', 'bang voor buitenlanders', 'een hekel hebbend aan buitenlanders', 'slagzin', 'kreet', 'jong persoon met een goede baan', 'zich kunnen redden', 'zes', 'zeven', 'nadenken over', 'overleggen met', 'betrekking hebben op', 'gaan over', 'in elk geval', 'wat er verder ook van te zeggen is', 'of dit nu werkelijk het geval is', 'aan de orde is of niet', 'van de kant van', 'van', 'van de kant van', 'door', 'zo', 'zo', 'zomaar', 'gewoon', 'altijd', 'helemaal', 'ervoor zorgen', 'netjes', 'precies', 'nauwkeurig', 'zo snel mogelijk', 'dit', 'dat', 'zoiets', 'dit betekent dus', 'het gevolg hiervan is dat', 'iets slechts aanbieden', 'alleen maar omdat er niets beters is', 'slecht voorstel']

informal_list_final.extend(informal_list_extended)

#location of sound files for all of the VA prompts, obtained through gTTS
sound_path = "C:\\LASnew\\Year 5\\Semester 2\\pipeline V2\\va_prompt_sound_files"

level_1 = 1
level_5 = 5
level_adaptable = "adaptable"

formality_level = "adaptable"                 # enter: level_1 / level_5 / level_adaptable
adaptive = True                          # only True if formality_level == level_adaptable
participant_number = 2                   # fill out participant number

#set starting point of adaptable formality level
formality_lvl_adaptable_set = 3

#set conditions for logging
logging.basicConfig(
    level = logging.DEBUG,                            
    format = '{asctime} {levelname:<8} {message}',    
    style = '{',
    filename ='{x}_participant_{y}.log'.format(x=__file__[:-3], y=str(participant_number)),                 #logging to a file
    filemode = 'a'                                                                                          #append to file
      )  

#pre-load model for similarity analysis
dataset = "nl_core_news_lg"
nlp = spacy.load(dataset)

#set values for boosting algorithm
columns = ["similarity", "sentiment", "word_list_match", "formality"]

data = np.array([[0.37462300541815835, 5, -28.57142857142857, 3], 
[0.9145499537253722, 5, -16.000000000000004, 2],
[0.744433403113071, 4, -22.22222222222222, 2],
[0.7146834622919648, 1, -19.999999999999993, 3],
[0.6794253818435918, 1, -20.0, 3],
[0.7299220257940099, 3, -32.55813953488372, 2],
[0.4693985702645436, 3, -17.857142857142858, 3],
[0.6560516722751073, 3, -29.166666666666664, 2],
[0.7383608902955988, 5, -22.22222222222223, 2],
[0.8066701833650148, 2, -26.66666666666666, 2],
[0.34693071756068233, 5, 0.0, 4],
[0.7715524254878443, 1, -11.11111111111111, 3],
[0.6847903520015255, 5, 0.0, 1],
[0.5985181855479061, 5, -42.85714285714286, 1],
[0.7832258659278982, 3, -28.57142857142857, 1],
[0.8411852260008026, 4, -28.20512820512821, 1],
[0.7409412377986161, 5, -33.33333333333333, 1],
[0.6905456269152539, 3, -40.74074074074074, 2],
[0.7165957785904983, 4, -24.13793103448276, 1],
[0.8236326279937982, 4, -35.13513513513513, 2],
[0.7173539284162785, 5, -23.529411764705884, 1],
[0.6841809972887788, 5, -18.18181818181818, 1],
[0.8575847548450269, 3, -38.46153846153846, 1],
[0.7985742900869318, 5, -7.142857142857142, 1],
[0.8678257303989828, 1, -16.666666666666664, 3],
[0.6375690419862133, 1, -30.0, 4],
[0.7285786690489167, 5, -7.692307692307692, 4],
[0.6671217481908295, 3, -15.384615384615385, 5],
[0.6949157209166618, 4, 12.5, 5],
[0.5663114225162145, 3, -22.22222222222222, 4],
[0.5752643264066204, 3, -31.578947368421048, 4],
[0.5007589583083656, 4, -8.695652173913045, 5],
[0.36134270839987215, 3, 9.523809523809526, 5],
[0.622488632570044, 5, -15.789473684210526, 5],
[0.45634490122511984, 4, -26.31578947368421, 3],
[0.6334067141837466, 3, -20, 5],
[0.2803944307864821, 5, -25, 4],
[0.7992627875876848, 5, -25, 4]])


df = pd.DataFrame(data=data, columns=columns)

        #divide data for training set
X_train = df[["similarity", "sentiment", "word_list_match"]]

y_train = df["formality"]

        #define names for testing columns
test_columns = ["similarity", "sentiment", "word_list_match"]

        #initialize boosting
bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3), n_estimators=300, learning_rate=1
)

        #train the model
bdt_real.fit(X_train, y_train)

#define function for similarity analysis
def similarity_out(va_prompt, user_input):
    """ Calculate similarity score between two string inputs 

            Parameters:
            va_prompt: a string
            user_input: a string

            Returns:
            The similarity score between two string inputs
    """

    va_ready = nlp(va_prompt)
    user_ready =  nlp(user_input)
    sim_score = va_ready.similarity(user_ready)

    return sim_score


#pre-load models for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

#define function for sentiment analysis
def sentiment_out(user_input):
    """ Calculate sentiment score of single string input 
    
            Parameters:
            user_input: a string

            Returns:
            The sentiment score of a string input
    """

    tokens = tokenizer.encode(user_input, return_tensors='pt')
    result_sentiment = sentiment_model(tokens)
    sent_score = int(torch.argmax(result_sentiment.logits))+1

    return sent_score


#define function for obtaining data for keyword matching
def input_list_count(informallist, formallist, user_input):

    """compare user input to 2 lists"""

    user_input_list = (user_input.split(" "))
    informal_count = 0
    formal_count = 0
    
    for count in user_input_list:
        if count in informallist:
           informal_count += 1
        if count in formallist:
            formal_count += 1
    # print("informal count of user input:", informal_count,"| formal count for user input:", formal_count)
    return formal_count, informal_count


#define function for formal/informal list keyword matching score
def obtain_match_score(user_input, match_count_formal, match_count_informal):
    """ Calculates the score of the relation between the amount of occurrences of formal versus informal words in the input, ranging between 100 and -100
            
            Parameters:
            user_input: a string
            match_count_formal: an integer
            match_count_informal: an integer

            Returns:
            A float between 100 and -100 that represents the relation between the amount of formal matches versus the amount of informal matches in the input
    """
    len_user_input = len(user_input.split(" "))
    formal_score = (match_count_formal/len_user_input)*100
    informal_score = (match_count_informal/len_user_input)*100
    formality_lvl_score = 0 + formal_score - informal_score
    return formality_lvl_score

#define function for pitch analysis
def obtain_pitch(file_path):
    """
    Parameters:
            user_input: a string
            match_count_formal: an integer
            match_count_informal: an integer

            Returns:
            The standard deviations of the pitch values
    """
    snd = parselmouth.Sound(file_path)

    pitch = snd.to_pitch()

    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values!=0]
    return pitch_values


#load model for vosk
model = Model(r"C:\\LASnew\\Year 5\\Semester 2\\vosk\\vosk-model-small-nl-0.22\\vosk-model-small-nl-0.22")                            #smaller model
# model = Model(r"C:\\LASnew\\Year 5\Semester 2\\Pipeline V9 on\\vosk-model-nl-spraakherkenning-0.6\\vosk-model-nl-spraakherkenning-0.6") #larger model
recognizer = KaldiRecognizer(model, 16000)

#start audio stream
cap = pyaudio.PyAudio()
#stream = cap.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)  #original
#stream = cap.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=2048)  #moved within for-loop
# stream.start_stream()

#obtain stt transcript per item
temp_storage = []
#complete_storage = []
score_list_total = []
prompt_counter = 0

if adaptive == False:
    if formality_level == 1:
        va_prompt_list = holiday_form_lvl_1
    if formality_level == 5:
        va_prompt_list = holiday_form_lvl_5
    for va_prompt in va_prompt_list:
        prompt_counter += 1

        playsound.playsound(r".\\va_prompt_sound_files\\prompt_{x}_formality_level_{y}.mp3".format(x=str(prompt_counter), y=str(formality_level)))
        logging.info("finish playsound")
        stream = cap.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
        stream.start_stream()
        logging.info("stream started")
        while True:
            #data = stream.read(4096)   #original
            data = stream.read(4096, exception_on_overflow = False)
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                temp_storage.append(json.loads(result)['text'])
                #if len(temp_storage) == 2 and temp_storage == ["", ""]:                     #original
                if len(temp_storage) == 1 and temp_storage == [""]:
                    print(temp_storage)
                    temp_storage.clear()
                else:
                    #if len(temp_storage) > 3 and temp_storage[-1: -3: -1] == ["", ""]:      #original, 20 second wait
                    #if len(temp_storage) > 2 and temp_storage[-1: -2: -1] == [""]:          #improved by 5 sec, 15 second wait
                    if len(temp_storage) > 1 and temp_storage[-1: -2: -1] == [""]:          #further improved by 5 seconds, 10 second wait
                        stream.close()
                        text_out = ' '.join(temp_storage)
                        logging.info(text_out)
                        temp_storage.clear()
                        #complete_storage.append(text_out)          #check if this line is actually neccessary
                        #print(text_out)
                        sim_score = similarity_out(va_prompt, text_out)
                        logging.info("similarity score prompt {x}: {y}".format(x=prompt_counter, y=sim_score))
                        sent_score = sentiment_out(text_out)
                        logging.info("sentiment score prompt {x}: {y}".format(x=prompt_counter, y=sent_score))
                        count_formal, count_informal = input_list_count(informal_list_final, formal_list_final, text_out)
                        formality_match_score = obtain_match_score(text_out, count_formal, count_informal)
                        logging.info("formality matching score prompt {x}: {y}".format(x=prompt_counter, y=formality_match_score))
                        test_data = np.array([[sim_score, sent_score, formality_match_score]])
                        df_test = pd.DataFrame(data=test_data, columns=test_columns)
                        X_test = df_test
                        label = bdt_real.staged_predict(X_test)
                        predictions = pd.DataFrame({"formality":label})
                        result_array = predictions["formality"].loc[predictions.index[0]]
                        result_boosting = int(result_array.item())
                        logging.info("boosting result prompt {x}: {y}".format(x=prompt_counter, y=result_boosting))
                        break
else:
    if adaptive == True:
        if formality_level == "adaptable":
            va_prompt_list = holiday_form_lvl_adaptable
            for diff_levels in va_prompt_list:
                prompt_counter += 1
                
                playsound.playsound(r".\\va_prompt_sound_files\\prompt_{x}_formality_level_{y}.mp3".format(x=str(prompt_counter), y=str(formality_lvl_adaptable_set)))

                stream = cap.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
                stream.start_stream()
                while True:
                    #data = stream.read(4096)  #original
                    data = stream.read(4096, exception_on_overflow = False)
                    if recognizer.AcceptWaveform(data):
                        result = recognizer.Result()
                        temp_storage.append(json.loads(result)['text'])
                        #if len(temp_storage) == 2 and temp_storage == ["", ""]:        #original
                        if len(temp_storage) == 1 and temp_storage == [""]:
                            print(temp_storage)
                            temp_storage.clear()
                        else:
                            #if len(temp_storage) > 3 and temp_storage[-1: -3: -1] == ["", ""]:      #original, 20 second wait
                            #if len(temp_storage) > 2 and temp_storage[-1: -2: -1] == [""]:          #improved by 5 sec, 15 second wait
                            if len(temp_storage) > 1 and temp_storage[-1: -2: -1] == [""]:          #further improved by 5 seconds, 10 second wait
                                stream.close()
                                text_out = ' '.join(temp_storage)
                                logging.info(text_out)
                                temp_storage.clear()
                                #complete_storage.append(text_out)          #check if this line is actually neccessary
                                #print(text_out)
                                sim_score = similarity_out(diff_levels[formality_lvl_adaptable_set-1], text_out)
                                logging.info("similarity score prompt {x}: {y}".format(x=prompt_counter, y=sim_score))
                                sent_score = sentiment_out(text_out)
                                logging.info("sentiment score prompt {x}: {y}".format(x=prompt_counter, y=sent_score))
                                count_formal, count_informal = input_list_count(informal_list_final, formal_list_final, text_out)
                                formality_match_score = obtain_match_score(text_out, count_formal, count_informal)
                                logging.info("formality matching score prompt {x}: {y}".format(x=prompt_counter, y=formality_match_score))
                                test_data = np.array([[sim_score, sent_score, formality_match_score]])
                                df_test = pd.DataFrame(data=test_data, columns=test_columns)
                                X_test = df_test
                                label = bdt_real.staged_predict(X_test)
                                predictions = pd.DataFrame({"formality":label})
                                result_array = predictions["formality"].loc[predictions.index[0]]
                                result_boosting = int(result_array.item())
                                if result_boosting > formality_lvl_adaptable_set:
                                    formality_lvl_adaptable_set += 1

                                if result_boosting == formality_lvl_adaptable_set:
                                    formality_lvl_adaptable_set = formality_lvl_adaptable_set

                                if result_boosting < formality_lvl_adaptable_set:
                                    formality_lvl_adaptable_set -= 1
                                logging.info("boosting result prompt {x}: {y}".format(x=prompt_counter, y=result_boosting))
                                logging.info("new formality level {x}: {y}".format(x=prompt_counter, y=formality_lvl_adaptable_set))
                                break

