"""
Bridges farmer vocabulary to the technical language of agricultural guides.

Two gaps are handled:
  1. Igbo terms used by farmers in southeastern Nigeria
  2. English vernacular that differs from IITA/FAO phrasing

Diacritics are stripped before matching, so a farmer typing "akwukwo" on a
plain phone keyboard still matches "akwụkwọ".
"""

import unicodedata


def normalise(text):
    """Lowercase and strip diacritics so ụ->u, ọ->o, ị->i."""
    text = text.lower()
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


# --- Igbo terms -> technical English used in the guides ---
IGBO_TERMS = {
    # plant and parts
    "akpu": ["cassava"],
    "jiakpu": ["cassava"],
    "osisi akpu": ["cassava plant", "stem"],
    "mgborogwu": ["storage roots", "tuberous roots"],
    "akwukwo": ["leaf", "leaves"],
    "opi akpu": ["shoot tip", "growing tip"],
    "elu akpu": ["shoot tip", "growing tip"],
    "okwu akpu": ["stem", "stems"],
    "ogbe akpu": ["stem", "stems"],

    # colour and appearance
    "ocha": ["white", "whitish"],
    "edo": ["yellow", "chlorotic", "chlorosis"],
    "edo edo": ["yellow", "yellowing", "chlorotic", "chlorosis"],
    "nchara": ["brown", "brownish"],
    "oji": ["black", "dark"],
    "ntupo": ["spots", "spotted", "chlorotic spots"],
    "ntupo-ntupo": ["spots", "spotted", "chlorotic spots"],
    "icha mfe": ["pale", "chlorotic", "discoloration"],

    # symptoms
    "ikpocha": ["drying up", "dieback"],
    "iru ite": ["rot", "root rot", "decay"],
    "ila n'iyi": ["rot", "decay", "deteriorating"],
    "inwu": ["dying", "dieback", "plant death"],
    "ikpo nku": ["wilting", "drying up"],
    "ikpo nro": ["wilting"],
    "oghere": ["holes", "feeding damage", "defoliation"],
    "ikpa kpa": ["distorted", "crinkled", "mosaic", "deformed"],
    "ikpoko onu": ["bunchy top", "clumping", "shortened internodes"],
    "mkpirisi": ["stunted growth", "reduced growth"],
    "ikpa ami": ["stunted growth"],
    "nta": ["small", "reduced leaf size"],

    # pests and disease
    "aruru": ["pest", "insect"],
    "ahuhu": ["pest", "insect", "mealybug", "mite"],
    "ikpuru": ["caterpillar", "larvae", "feeding damage"],
    "oria": ["disease", "infection"],
    "nsi": ["blight", "rot", "necrosis"],
}

# --- English farmer vernacular -> technical guide language ---
ENGLISH_TERMS = {
    "curling": ["bunchy top", "clumping", "distorted", "deformed", "crinkled"],
    "curled": ["bunchy top", "clumping", "distorted", "deformed"],
    "curl": ["bunchy top", "clumping", "distorted"],
    "bunching": ["bunchy top", "clumping", "shortened internodes"],
    "bunched": ["bunchy top", "clumping"],
    "bushy": ["bunchy top", "clumping", "shoot tip"],
    "twisted": ["distorted", "deformed"],
    "wrinkling": ["distorted", "crinkled"],
    "shrivelled": ["wilting", "drying up"],
    "shriveled": ["wilting", "drying up"],

    "yellow": ["chlorotic", "chlorosis", "yellowing", "discoloration"],
    "yellowing": ["chlorotic", "chlorosis", "discoloration"],
    "pale": ["chlorotic", "chlorosis", "discoloration"],
    "brown spots": ["necrosis", "lesions", "leaf spot"],
    "black spots": ["necrosis", "lesions", "anthracnose"],
    "white powder": ["waxy material", "mealybug"],
    "white stuff": ["waxy material", "mealybug"],
    "sticky": ["honeydew", "sooty mold"],

    "top of the plant": ["shoot tip", "growing tip", "terminal shoot"],
    "top of plant": ["shoot tip", "growing tip", "terminal shoot"],
    "new leaves": ["young leaves", "shoot tip"],
    "young leaves": ["shoot tip", "terminal leaves"],
    "roots": ["storage roots", "tuberous roots"],

    "dying": ["dieback", "wilting", "necrosis"],
    "drying": ["dieback", "drying up", "wilting"],
    "stunted": ["stunted growth", "reduced growth"],
    "not growing": ["stunted growth", "reduced growth"],
    "small leaves": ["reduced leaf size", "stunted growth"],
    "falling off": ["defoliation", "leaf drop"],
    "rotting": ["root rot", "decay"],
    "holes in leaves": ["defoliation", "feeding damage"],

    "insects": ["pest", "mealybug", "mite", "whitefly"],
    "tiny insects": ["mealybug", "mite", "whitefly"],
    "flies": ["whitefly", "whiteflies"],
}

ALL_TERMS = {**IGBO_TERMS, **ENGLISH_TERMS}


def expand_query(question):
    """Append technical synonyms for any farmer terms found in the question."""
    q = normalise(question)
    additions = []

    for term, technical in ALL_TERMS.items():
        if normalise(term) in q:
            additions.extend(technical)

    if not additions:
        return question

    seen = set()
    unique = [t for t in additions if not (t in seen or seen.add(t))]
    return question + " " + " ".join(unique)
