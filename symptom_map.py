"""
Maps everyday farmer language to the technical vocabulary used in
agricultural guides (IITA, FAO, NAERLS). Used to expand queries before
retrieval, closing the gap between how farmers speak and how guides write.
"""

SYMPTOM_SYNONYMS = {
    # leaf shape / deformation
    "curling": ["bunchy top", "clumping", "distorted", "deformed", "crinkled"],
    "curled": ["bunchy top", "clumping", "distorted", "deformed"],
    "curl": ["bunchy top", "clumping", "distorted"],
    "twisted": ["distorted", "deformed", "malformed"],
    "bunched": ["bunchy top", "clumping", "shortened internodes"],
    "crumpled": ["distorted", "deformed", "crinkled"],
    "shrivelled": ["wilting", "drying up", "defoliation"],
    "shriveled": ["wilting", "drying up", "defoliation"],

    # colour changes
    "yellow": ["chlorotic", "chlorosis", "yellowing", "discoloration"],
    "yellowing": ["chlorotic", "chlorosis", "discoloration"],
    "pale": ["chlorotic", "chlorosis", "discoloration"],
    "brown spots": ["necrosis", "lesions", "leaf spot"],
    "black spots": ["necrosis", "lesions", "anthracnose"],
    "white powder": ["waxy material", "mealybug", "sooty mold"],
    "white stuff": ["waxy material", "mealybug"],

    # location on plant
    "top of the plant": ["shoot tip", "growing tip", "terminal shoot"],
    "top of plant": ["shoot tip", "growing tip", "terminal shoot"],
    "tip": ["shoot tip", "terminal shoot"],
    "new leaves": ["young leaves", "shoot tip"],
    "young leaves": ["shoot tip", "terminal leaves"],
    "stem": ["stems", "stalk"],
    "roots": ["storage roots", "tuberous roots"],

    # plant condition
    "dying": ["dieback", "wilting", "necrosis"],
    "drying": ["dieback", "drying up", "wilting"],
    "stunted": ["stunted growth", "reduced growth"],
    "not growing": ["stunted growth", "reduced growth"],
    "small leaves": ["reduced leaf size", "stunted growth"],
    "falling off": ["defoliation", "leaf drop"],
    "rotting": ["root rot", "decay"],
    "holes in leaves": ["defoliation", "feeding damage"],

    # pests as farmers describe them
    "insects": ["pest", "mealybug", "mite", "whitefly"],
    "tiny insects": ["mealybug", "mite", "whitefly"],
    "flies": ["whitefly", "whiteflies"],
    "sticky": ["honeydew", "sooty mold"],
}


def expand_query(question):
    """Append technical synonyms for any farmer terms found in the question."""
    q_lower = question.lower()
    additions = []

    for farmer_term, technical_terms in SYMPTOM_SYNONYMS.items():
        if farmer_term in q_lower:
            additions.extend(technical_terms)

    if not additions:
        return question

    # Remove duplicates while keeping order
    seen = set()
    unique = [t for t in additions if not (t in seen or seen.add(t))]

    return question + " " + " ".join(unique)
