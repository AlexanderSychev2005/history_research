# Errors of the constrained runs (first repeat)

FP = predicted, not in the gold; FN = in the gold, not predicted. Names are the gold labels.

## gold__llm-20b-paragraph-r1

**False positives**

- p005 Ełišē | ex:captiveIn | Gorgān
- p013 Łazar P‘arpec‘i | yago:influencedBy | Nerseh Kamsarakan
- p013 Nerseh Kamsarakan | schema:location | Armenia
- p032 T‘at‘ik | schema:memberOf | Łewondeank‘
- p013 Izdegerd | schema:location | Arzanene
- p014 Dagaris | ex:captiveIn | Huns
- p018 Grigor | ex:captiveIn | Byzantine Empire
- p019 Martyrdom of Mar Grigor | schema:about | Chosroes
- p019 Martyrdom of Mar Grigor | schema:about | Grigor

**Missed**

- p004 Gorgān | yago:neighbor | Caspian Sea
- p004 History of the Armenians | yago:creator | Łazar P‘arpec‘i
- p004 Łazar P‘arpec‘i | schema:location | Vałaršapat
- p004 Vahan Mamikonean | ex:opposed | Sasanian Empire
- p005 Ecclesiastical Histories | yago:creator | John of Ephesus
- p013 Aršamunik‘ | schema:location | Armenia
- p013 Aršawir Kamsarakan | ex:captiveIn | Gorgān
- p013 Łazar P‘arpec‘i | schema:location | Vałaršapat
- p013 Nerseh Kamsarakan | ex:relativeOf | Hrahat
- p013 Nerseh Kamsarakan | schema:location | Širak
- p018 Hephthalite Huns | yago:follows | Kidarite Huns
- p018 Sasanian Empire | ex:opposed | Kidarite Huns
- p024 Artašēs | ex:captiveIn | Khuzestan
- p024 T‘at‘ik | ex:captiveIn | Khuzestan
- p024 T‘at‘ik | schema:location | Basean
- p024 Vahan Mamikonean | ex:captiveIn | Khuzestan
- p024 Vahan Mamikonean | ex:relativeOf | Artašēs
- p024 Vahan Mamikonean | ex:relativeOf | Vasak
- p024 Vasak | ex:captiveIn | Khuzestan
- p024 Vasak | ex:relativeOf | Artašēs
- p013 Arzanene | schema:location | Persarmenia
- p013 Izdegerd | ex:captiveIn | Byzantine Empire
- p014 Dagaris | schema:location | Armenia
- p018 Grigor | schema:location | Sasanian Empire
- p018 Grigor | schema:worksFor | Byzantine Empire
- p019 Grigor | schema:location | Constantinople
- p019 Grigor | schema:memberOf | Church of the East

## gold__llm-20b-sentence-r1

**False positives**

- p004 Gorgān | schema:location | Alborz
- p004 Łazar P‘arpec‘i | schema:worksFor | Vahan Mamikonean
- p004 Vałaršapat | yago:partOf | Vałaršapat
- p005 Robert Thomson | yago:influencedBy | Akinean
- p005 Robert Thomson | yago:influencedBy | Kiwlēsērean
- p013 Nerseh Kamsarakan | schema:location | Aršamunik‘
- p013 Nerseh Kamsarakan | schema:location | Armenia
- p018 Hephthalite Huns | ex:opposed | Kidarite Huns
- p018 Pērōz | ex:opposed | Kidarite Huns
- p018 Yazdgerd II | ex:opposed | Kidarite Huns
- p032 T‘at‘ik | schema:location | Asorestan
- p013 Hormizd | schema:location | Arzanene
- p014 Dagaris | ex:captiveIn | Sasanian Empire
- p018 Justinian | yago:influencedBy | Grigor
- p019 Martyrdom of Mar Grigor | schema:about | Chosroes
- p019 Martyrdom of Mar Grigor | schema:about | Grigor

**Missed**

- p004 Vahan Mamikonean | ex:opposed | Sasanian Empire
- p013 Aršamunik‘ | schema:location | Armenia
- p013 Aršawir Kamsarakan | ex:captiveIn | Gorgān
- p013 Łazar P‘arpec‘i | schema:location | Vałaršapat
- p013 Nerseh Kamsarakan | ex:relativeOf | Hrahat
- p018 Hephthalite Huns | yago:follows | Kidarite Huns
- p018 Sasanian Empire | ex:opposed | Kidarite Huns
- p024 Artašēs | ex:captiveIn | Khuzestan
- p024 T‘at‘ik | ex:captiveIn | Khuzestan
- p024 Vahan Mamikonean | ex:captiveIn | Khuzestan
- p024 Vasak | ex:captiveIn | Khuzestan
- p032 T‘at‘ik | schema:deathPlace | Asorestan
- p004 Damascius | schema:location | Syria
- p004 Diogenes | schema:location | Phoenicia
- p004 Eulamius | schema:location | Phrygia
- p004 Hermes | schema:location | Phoenicia
- p004 Isidore | schema:location | Gaza
- p004 Priscian | schema:location | Lydia
- p004 Simplicius | schema:location | Cilicia
- p018 Grigor | schema:location | Sasanian Empire
- p018 Grigor | schema:worksFor | Justinian
- p018 Grigor | schema:worksFor | Byzantine Empire
