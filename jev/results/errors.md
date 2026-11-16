# Jev pipeline: errors against dataset/gold

Written by `python jev/score.py --errors` from `jev/pred/`.

## End to end (Jev finds the mentions)

### alford2026

Mentions missed: Armenian rebellion (p001), Armenian rebellion (p003), John of Ephesus (p005)

Mentions that are not in the gold: John (p005), Ephesus (p005), Sasanian (p032), Mamikonean (p032)

| mention | times | linked to | gold |
|---|---|---|---|
| Nerseh | 7 | unlinked | yago:Nerses_Kamsarakan |
| Vahan | 3 | yago:Vahan_II_Mamikonian | yago:Vahan_I_Mamikonian |
| Ašuša | 2 | yago:Arshusha_II | no entry in YAGO |
| Basean | 2 | yago:Phasiane_historical_region | no entry in YAGO |
| Vahan Mamikonean | 2 | yago:Vahan_II_Mamikonian | yago:Vahan_I_Mamikonian |
| Armenia | 1 | unlinked | yago:Armenia |
| Nerseh Kamsarakan | 1 | unlinked | yago:Nerses_Kamsarakan |
| Siwnik‘ | 1 | yago:Syunik_historical_province | yago:Syunik_Province |
| Tokharistan | 1 | yago:Tokharistan | no entry in YAGO |

| fact that is not in the gold | paragraph | p |
|---|---|---|
| Łazar — schema:worksFor → Vahan | p004 | 0.95 |
| Robert Thomson — yago:influencedBy → Akinean | p005 | 0.77 |
| Robert Thomson — yago:influencedBy → Kiwlēsērean | p005 | 0.74 |
| Ecclesiastical Histories — yago:creator → John | p005 | 0.91 |
| John — schema:location → Ephesus | p005 | 0.83 |
| Łazar — yago:influencedBy → Nerseh | p013 | 0.81 |
| Pērōz — ex:opposed → Kidarites | p018 | 0.77 |
| Tokharistan — yago:neighbor → Hindu Kush | p018 | 0.77 |
| Kamsarakan family — ex:captiveIn → Khuzestan | p024 | 0.76 |
| Łazar — ex:captiveIn → Siwnik‘ | p024 | 0.91 |
| Vasak Siwni — ex:opposed → Łazar | p024 | 0.83 |

Gold facts missed: Armenian rebellion — schema:location → Awarayr (p003); Ecclesiastical Histories — yago:creator → John of Ephesus (p005); Aršamunik‘ — schema:location → Armenia (p013)

### nechaeva2017

Mentions missed: Mar Grigor (p019)

Mentions that are not in the gold: Amidan (p013), Hun (p014), Grigor (p018), Grigor (p019)

| mention | times | linked to | gold |
|---|---|---|---|
| Domitziolus | 2 | yago:Domnentiolus | no entry in YAGO |
| Pseudo-Zacharias | 2 | yago:Pseudo-Zacharias_Rhetor | no entry in YAGO |
| Armenia | 1 | unlinked | yago:Armenia |
| Callinicum | 1 | unlinked | yago:Raqqa |
| Domnentiolus | 1 | yago:Domnentiolus | no entry in YAGO |
| Dorotheus | 1 | yago:Dorotheus_magister_militum | no entry in YAGO |
| Gaza | 1 | unlinked | yago:Gaza_City |
| Isidore | 1 | yago:Isidore_of_Alexandria | no entry in YAGO |
| Persarmenia | 1 | yago:Sasanian_Armenia | no entry in YAGO |
| Roman Empire | 1 | yago:Roman_Empire | yago:Byzantine_Empire |
| Syria | 1 | yago:Syria | no entry in YAGO |

| fact that is not in the gold | paragraph | p |
|---|---|---|
| Hormizd — schema:location → Arzanene | p013 | 0.92 |
| Hormizd — schema:location → Persarmenia | p013 | 0.76 |
| Pseudo-Zacharias — yago:influencedBy → Amidan | p013 | 0.74 |
| Dagaris — ex:captiveIn → Hun | p014 | 0.93 |
| Dagaris — schema:worksFor → Dorotheus | p014 | 0.79 |
| Dagaris — schema:worksFor → Sittas | p014 | 0.78 |
| Grigor — schema:worksFor → Persia | p018 | 0.92 |
| Zabergān — schema:worksFor → Chosroes | p019 | 0.79 |
| Grigor — schema:location → Persia | p019 | 0.78 |
| Martyrdom — schema:about → Grigor | p019 | 0.72 |

Gold facts missed: Diogenes — schema:location → Sasanian Empire (p004); Hermes — schema:location → Phoenicia (p004); Grigor — schema:location → Sasanian Empire (p018); Grigor — schema:worksFor → Justinian (p018)

## On the gold mentions (stages 2 and 3 alone)

### alford2026

Mentions missed: none

Mentions that are not in the gold: none

| mention | times | linked to | gold |
|---|---|---|---|
| Nerseh | 7 | unlinked | yago:Nerses_Kamsarakan |
| Vahan | 3 | yago:Vahan_II_Mamikonian | yago:Vahan_I_Mamikonian |
| Ašuša | 2 | yago:Arshusha_II | no entry in YAGO |
| Basean | 2 | yago:Phasiane_historical_region | no entry in YAGO |
| Vahan Mamikonean | 2 | yago:Vahan_II_Mamikonian | yago:Vahan_I_Mamikonian |
| Armenia | 1 | unlinked | yago:Armenia |
| Nerseh Kamsarakan | 1 | unlinked | yago:Nerses_Kamsarakan |
| Siwnik‘ | 1 | yago:Syunik_historical_province | yago:Syunik_Province |
| Tokharistan | 1 | yago:Tokharistan | no entry in YAGO |

| fact that is not in the gold | paragraph | p |
|---|---|---|
| Armenian rebellion — yago:winner → Yazdgerd II | p001 | 0.87 |
| Armenian rebellion — yago:hasCause → Yazdgerd II | p003 | 0.93 |
| Łazar — schema:worksFor → Vahan | p004 | 0.95 |
| Robert Thomson — yago:influencedBy → Akinean | p005 | 0.77 |
| Robert Thomson — yago:influencedBy → Kiwlēsērean | p005 | 0.74 |
| Łazar — yago:influencedBy → Nerseh | p013 | 0.81 |
| Pērōz — ex:opposed → Kidarites | p018 | 0.77 |
| Tokharistan — yago:neighbor → Hindu Kush | p018 | 0.77 |
| Kamsarakan family — ex:captiveIn → Khuzestan | p024 | 0.76 |
| Łazar — ex:captiveIn → Siwnik‘ | p024 | 0.91 |
| Vasak Siwni — ex:opposed → Łazar | p024 | 0.83 |

Gold facts missed: Armenian rebellion — schema:location → Awarayr (p003); Aršamunik‘ — schema:location → Armenia (p013)

### nechaeva2017

Mentions missed: none

Mentions that are not in the gold: none

| mention | times | linked to | gold |
|---|---|---|---|
| Domitziolus | 2 | yago:Domnentiolus | no entry in YAGO |
| Pseudo-Zacharias | 2 | yago:Pseudo-Zacharias_Rhetor | no entry in YAGO |
| Armenia | 1 | unlinked | yago:Armenia |
| Callinicum | 1 | unlinked | yago:Raqqa |
| Domnentiolus | 1 | yago:Domnentiolus | no entry in YAGO |
| Dorotheus | 1 | yago:Dorotheus_magister_militum | no entry in YAGO |
| Gaza | 1 | unlinked | yago:Gaza_City |
| Isidore | 1 | yago:Isidore_of_Alexandria | no entry in YAGO |
| Persarmenia | 1 | yago:Sasanian_Armenia | no entry in YAGO |
| Roman Empire | 1 | yago:Roman_Empire | yago:Byzantine_Empire |
| Syria | 1 | yago:Syria | no entry in YAGO |

| fact that is not in the gold | paragraph | p |
|---|---|---|
| Hormizd — schema:location → Arzanene | p013 | 0.92 |
| Hormizd — schema:location → Persarmenia | p013 | 0.76 |
| Dagaris — schema:worksFor → Dorotheus | p014 | 0.79 |
| Dagaris — schema:worksFor → Sittas | p014 | 0.78 |
| Grigor — schema:worksFor → Persia | p018 | 0.92 |
| Zabergān — schema:worksFor → Chosroes | p019 | 0.79 |
| Grigor — schema:location → Persia | p019 | 0.78 |
| Martyrdom — schema:about → Grigor | p019 | 0.72 |

Gold facts missed: Diogenes — schema:location → Sasanian Empire (p004); Hermes — schema:location → Phoenicia (p004); Grigor — schema:location → Sasanian Empire (p018); Grigor — schema:worksFor → Justinian (p018)
