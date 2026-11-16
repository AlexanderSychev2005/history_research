"""Every question the pipeline asks Jev, and every threshold it reads the answers with, in one place.

Rules followed (TypeSafe guide for Jev 1.14):
* one observable judgment per question; the subject is named in the question, never left to an id;
* a Choice never has a "none" option: a companion Noul asks whether any option fits;
* option names say what they mean (the model sees them), and the false case of a Noul is a concrete situation;
* Jev leans to the first option of a Choice, so every Choice is asked in two option orders and averaged (`both_orders`);
* code decides what code can decide (which relations two types allow, which words are capitalised);
* no example in a question comes from the papers of dataset/ (Venetian ships, Bank of England, Hugh of Lincoln), so
  that a question cannot have been fitted to a name it will meet there.
"""

# Thresholds. None was fitted to the gold: 0.5 is "more likely than not", 0.7 is the guide's starting point for a
# clear yes and is used where a wrong yes ends up in the knowledge graph. jev/score.py --sweep shows the others.
T_NAME = 0.5        # a candidate is a name
T_SAME = 0.5        # two written forms are one entity
T_LINK = 0.7        # a YAGO entry is the entity
T_FACT = 0.7        # a relation is stated

KINDS = {  # kind -> (description for Jev, type written to the output, YAGO classes for the relation stage)
    "person": ("one particular human being", "person", ("schema:Person",)),
    "place": ("a city, region, province, mountain range, sea or other geographic place", "location", ("schema:Place",)),
    "people_or_state": ("a people, tribe, empire, kingdom or state, such as the Franks, the Mongols or the Ottoman Empire",
                        "organization", ("schema:Place", "schema:Organization")),
    "group": ("a family, clan, dynasty, church, army or other institution", "organization", ("schema:Organization",)),
    "written_work": ("a book, history, chronicle or other text", "written work", ("schema:CreativeWork",)),
    "event": ("a battle, war, rebellion, treaty or other event", "event", ("schema:Event",)),
}


def both_orders(name, question):
    """A Choice as two questions, the second with the options reversed; read with `choice_probs`."""
    crit = question["criteria"]
    return {name: question, name + "~r": {**question, "criteria": dict(reversed(list(crit.items())))}}


def choice_probs(answers, name):
    a, b = answers[name]["probabilities"], answers[name + "~r"]["probabilities"]
    return {k: (a[k] + b[k]) / 2 for k in a}


# ---------------------------------------------------------------- stage 1: mentions

def is_name(text):
    return {"type": "noul",
            "instructions": f"In `sentence`, is “{text}” used as the proper name of a particular person, place, people, "
                            "state, family, institution, written work or event?",
            "criteria": {"true": f"“{text}” is a name: it stands for one particular person, place, people, state, family, "
                                 "institution, written work or event.",
                         "false": f"“{text}” is an adjective describing the noun after it (as in “Venetian ships”), a "
                                  "title or office, a religion or language, an ordinary word that is capitalised only "
                                  "because it starts the sentence, or part of a source citation."}}


def kind(text):
    return {"type": "choice", "instructions": f"What does “{text}” name in `sentence`?",
            "criteria": {k: v[0] for k, v in KINDS.items()}}


def name_boundary(options):
    """options: the suffixes of a capitalised run, longest first (English King Henry II / King Henry II / Henry II).
    Which one is the person's name without titles?"""
    full, crit = options[0], {}
    for o in options:
        cut = full[:len(full) - len(o)].strip()
        crit[o] = (f"All of “{o}” is the person's name." if not cut else
                   f"The person's name is “{o}”. “{cut}” in front of it is a title, rank, office or nationality, "
                   "not part of the name.")
    return {"type": "choice", "instructions": f"In `sentence`, “{full}” refers to a person. Which words are the "
                                              "person's name itself?", "criteria": crit}


def of_reading(a, b):
    return {"type": "choice", "instructions": f"In `sentence`, how is “{a} of {b}” built?",
            "criteria": {"one_name": f"“{a}” alone is not a name: “{a} of {b}” is one name of an institution, a work "
                                     "or an event, like “Bank of England” or “Battle of Hastings”.",
                         "name_and_origin": f"“{a}” is the name of a person, and “{b}” is a separate place or group "
                                            "that the person comes from, rules or belongs to, like “Hugh of Lincoln”."}}


def with_head_noun(name, noun):
    return {"type": "noul",
            "instructions": f"In `sentence`, is “{name} {noun}” as a whole the established name of one particular "
                            f"historical {noun}?",
            "criteria": {"true": f"“{name} {noun}” names one specific {noun} that the text treats as a known thing, the "
                                 "way “French Revolution” names an event.",
                         "false": f"“{noun}” is an ordinary noun and “{name}” only says which or whose one is meant; the "
                                  "two words are not a name together."}}


# ---------------------------------------------------------------- stage 2: entities and links

def same_entity(short, long_):
    return {"type": "noul",
            "instructions": f"In `paragraph`, do “{short}” and “{long_}” refer to the same one?",
            "criteria": {"true": f"“{short}” is a shorter way of writing “{long_}”: both stand for the same person, "
                                 "place, group or work.",
                         "false": f"“{short}” and “{long_}” are two different persons, places, groups or works that "
                                  "share part of a name."}}


def other_name(first, second):
    return {"type": "noul",
            "instructions": f"In `sentence`, is “{second}” given as another name or spelling of “{first}”?",
            "criteria": {"true": f"The brackets after “{first}” give “{second}” as a second name, a variant spelling or "
                                 "the modern or Latin name of the same person or place.",
                         "false": f"“{second}” is a different person or place that the brackets mention."}}


def link_any(text):
    """The companion Noul of `link_which`. An entry for the same place under its modern name counts, as
    dataset/README.md asks (its "approx" links)."""
    return {"type": "noul",
            "instructions": f"Is one of the `candidates` the entity that “{text}” refers to in `context`?",
            "criteria": {"true": f"One candidate is who or what “{text}” is in the context: the same person, people, "
                                 "state, institution or work, or the same place, also when the entry describes that "
                                 "place under its modern name or as today's city or province.",
                         "false": f"No candidate is “{text}” itself. The candidates are other persons with the same "
                                  "name, places elsewhere, or topics that are only related to it, such as a battle "
                                  "fought at the place or a book about the person."}}


def link_which(text, candidates):
    """candidates: [(title, description)] -> a Choice whose option names are the titles."""
    return {"type": "choice",
            "instructions": f"Which knowledge-base entry is the entity that “{text}” refers to in `context`?",
            "criteria": {t: (d or "no description") for t, d in candidates}}


def link_is(text, title, description):
    what = f"“{title}”" + (f" ({description})" if description else "")
    return {"type": "noul",
            "instructions": f"In `context`, is “{text}” the entity described by the knowledge-base entry {what}?",
            "criteria": {"true": f"“{text}” in the context and the entry {what} are one and the same; for a place, the "
                                 "entry may describe the same place under its modern name or as today's city or province.",
                         "false": f"The entry {what} is someone or something else than “{text}” in the context: another "
                                  "person of that name, a place elsewhere, or a topic that is only related to it."}}


# --- profiles: the same questions about the entity in the text and about a knowledge-base entry; code compares

def _ordinal(n):
    return f"{n}{'th' if 10 < n % 100 < 14 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# the 6th century BC to the 21st century; CENTURIES[6] is the 1st century
CENTURIES = [f"{_ordinal(n)} century BC" for n in range(6, 0, -1)] + [f"{_ordinal(n)} century" for n in range(1, 22)]


def century_of(text):
    return {"type": "choice", "instructions": f"In which century did “{text}” live, exist or take place, going by "
                                              "`context`?", "criteria": {c: None for c in CENTURIES}}


def dated(text):
    return {"type": "noul",
            "instructions": f"Does `context` show in which century “{text}” lived, existed or took place?",
            "criteria": {"true": f"Dates, events or contemporaries in the context place “{text}” in a century.",
                         "false": f"Nothing in the context places “{text}” in time."}}


def entry_kind():
    return {"type": "choice", "instructions": "What is the subject of `entry`?",
            "criteria": {k: v[0] for k, v in KINDS.items()}}


def entry_century():
    return {"type": "choice", "instructions": "In which century did the subject of `entry` live, exist or take place?",
            "criteria": {c: None for c in CENTURIES}}


def entry_dated():
    return {"type": "noul", "instructions": "Does `entry` show in which century its subject lived or took place?",
            "criteria": {"true": "The entry gives years or a century for its subject.",
                         "false": "The entry gives no dates."}}


def pick_article(text, titles):
    """One round of the tournament over article titles alone (up to 255 options)."""
    return {"type": "choice",
            "instructions": f"Which of these encyclopedia articles is about “{text}” as `context` uses the name?",
            "criteria": {t: None for t in titles}}


# ---------------------------------------------------------------- stage 3: relations

# How each relation is put to Jev: {s} and {o} are the two names. Follows dataset/relations.json and the conventions
# of dataset/README.md; where two relations were confused, the wording names the case that separates them.
RELATIONS = {
    "yago:creator": "{o} wrote or made {s}",
    "schema:location": "{s} lived in, was based in, was sent to or lies in {o}, or is called “{s} of {o}”",
    "schema:about": "the work {s} is about {o}",
    "yago:influencedBy": "{s} was influenced by {o}",
    "yago:leader": "{o} led or ruled {s}",
    "yago:winner": "{o} won {s}",
    "yago:participant": "{o} took part in {s}",
    "yago:hasCause": "{s} was caused by {o}",
    "yago:neighbor": "{s} and {o} border each other",
    "yago:studentOf": "{s} studied under {o}",
    "schema:worksFor": "{s} served, worked for or held office under {o}",
    "schema:birthPlace": "{s} was born in {o}",
    "yago:hasMother": "{o} is the mother of {s}",
    "yago:hasFather": "{o} is the father of {s}",
    "schema:deathPlace": "{s} died or was killed in {o}",
    "yago:burialPlace": "{s} was buried in {o}",
    "schema:spouse": "{s} and {o} were married to each other",
    "schema:memberOf": "{s} belonged to the family, dynasty, church or institution {o}",
    "yago:partOf": "{s} is a part or subdivision of {o}",
    "yago:follows": "{s} came after and replaced {o}",
    "ex:relativeOf": "{s} and {o} are relatives other than parent and child or husband and wife: brothers, uncle and "
                     "nephew, cousins or in-laws",
    "ex:captiveIn": "{s} was taken prisoner by {o}, or was deported to or held captive in {o}",
    "ex:opposed": "{s} fought, rebelled against or persecuted {o}",
}
SYMMETRIC = {"schema:spouse", "yago:neighbor", "ex:relativeOf"}   # asked once; subject = the name that comes first
SAME_CLASS = {"yago:partOf", "yago:follows"}                      # only between two things of one kind


def option_name(rel, s, o):
    return f"{rel.split(':')[1]}({s}, {o})"


def related(a, b):
    return {"type": "noul",
            "instructions": f"Does `sentence` say how “{a}” and “{b}” are connected to each other?",
            "criteria": {"true": f"The sentence states a connection between “{a}” and “{b}” themselves: kinship, service, "
                                 "membership, captivity, living or lying in a place, conflict, authorship or succession.",
                         "false": f"“{a}” and “{b}” only appear in the same sentence. What is said about one of them is "
                                  "not said about the other."}}


def which_relation(a, b, options):
    """options: [(rel, subject, object)]"""
    return {"type": "choice", "instructions": f"Which connection between “{a}” and “{b}” does `sentence` state?",
            "criteria": {option_name(r, s, o): RELATIONS[r].format(s=f"“{s}”", o=f"“{o}”") for r, s, o in options}}


def states(rel, s, o):
    claim = RELATIONS[rel].format(s=f"“{s}”", o=f"“{o}”")
    return {"type": "noul",
            "instructions": f"Does `sentence` say that {claim}?",
            "criteria": {"true": f"The sentence states or directly implies that {claim}.",
                         "false": f"The sentence does not say this about “{s}” and “{o}”: it says it about someone or "
                                  "something else, or it connects the two in another way."}}
