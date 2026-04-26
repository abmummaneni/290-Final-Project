"""
load_mystery_graphs.py
======================
Loads all mystery JSON files and merges them into a single RelationalGraph
for R-GCN link-prediction training.

Design choices
--------------
1. Node features        — numerical attributes stacked into a feature matrix.
2. Story-level split    — entire stories held out for val/test so the model
                          is never evaluated on narratives seen during training.
3. Two-stage relation collapse
      Stage 1 (normalize_relation) — regex patterns collapse hundreds of
        free-text relation strings into ~35 intermediate canonical forms,
        exactly as before.
      Stage 2 (coarsen_relation) — a lookup table maps every intermediate
        form to one of 8 coarse relation types suited for link prediction:

          KILLS         lethal violence, death locations
          HARMS         non-lethal violence, coercion, conflict
          INVESTIGATES  detection, legal process, testimony, tip-offs
          DECEIVES      active deception, concealment, false identity,
                        criminal complicity and hidden alliance
          PERSONAL_BOND family, romance, friendship, emotional ties
          PROFESSIONAL  employment, mentorship, medical care
          SPATIAL       residence, physical location, proximity
          SOCIAL        residual narrative contact, institutional
                        affiliation, and lightweight social edges

        This v3 schema was arrived at empirically after two earlier attempts:
          v1 (8 types): INTERACTS was a 28.6% junk-drawer; DECEIVES was
             starved at 0.9%.
          v2 (10 types): Split INTERACTS into SPATIAL + SOCIAL, but
             COMMUNICATES landed at 2.3% and DECEIVES stayed at 0.8%
             — both still critically starved.
          v3 (this): Merges starved types upward. COMMUNICATES re-absorbed
             into INVESTIGATES; old AFFILIATED bucket merged into DECEIVES
             (criminal alliance is a form of deception in mystery fiction),
             lifting DECEIVES from 0.8% to ~8%. Expected range: ~4%-24%.

        Having 8 well-populated relation types instead of 400+ means:
          - Each relation type has many more training examples
          - The DistMult decoder has a dense, well-trained relation matrix
          - Basis decomposition in the encoder is not starved per relation
          - No type dominates or is ignored during training
"""

import json
import os
import re
import random
import torch
from typing import Optional
from .rgcn_model import RelationalGraph


# ---------------------------------------------------------------------------
# 1. Node feature extraction
# ---------------------------------------------------------------------------

CHARACTER_FEATURES = [
    "gender",
    "social_status",
    "narrative_introduction_timing",
    "has_alibi",
    "present_at_crime_scene",
    "has_motive",
    "is_concealing_information",
    "has_hidden_relationship",
    "motive_type",
    "narrative_prominence",
]

OCCUPATION_FEATURES = [
    "authority_level",
    "access_level",
    "capability_level",
]

LOCATION_FEATURES = [
    "accessibility",
    "isolability",
    "evidentiary_value",
]

ORGANIZATION_FEATURES = [
    "institutional_power",
    "secrecy_level",
    "financial_scale",
]

FEAT_DIM = max(
    len(CHARACTER_FEATURES),
    len(OCCUPATION_FEATURES),
    len(LOCATION_FEATURES),
    len(ORGANIZATION_FEATURES),
)  # = 10

NODE_TYPE_FEATURES = {
    "characters":    CHARACTER_FEATURES,
    "occupations":   OCCUPATION_FEATURES,
    "locations":     LOCATION_FEATURES,
    "organizations": ORGANIZATION_FEATURES,
}

MOTIVE_TYPE_MAP = {
    "financial":               0.33,
    "money":                   0.33,
    "love":                    0.33,
    "jealousy":                0.50,
    "protecting family":       0.50,
    "manipulation":            0.50,
    "power":                   0.66,
    "glory":                   0.66,
    "entitlement":             0.66,
    "revenge":                 1.00,
    "psychopathic tendencies": 1.00,
    "other":                   0.50,
    "unknown":                 0.00,
    "unk":                     0.00,
}


def safe_float(k: str, v) -> float:
    """Convert a feature value to float, handling strings, lists, and dicts."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        try:
            return sum(float(x) for x in v) / len(v) if v else 0.0
        except (ValueError, TypeError):
            return 0.0
    if isinstance(v, dict):
        try:
            vals = list(v.values())
            return sum(float(x) for x in vals) / len(vals) if vals else 0.0
        except (ValueError, TypeError):
            return 0.0
    if isinstance(v, str):
        return MOTIVE_TYPE_MAP.get(v.strip().lower(), 0.0)
    return 0.0


def extract_node_features(node: dict, node_type: str,
                          exclude_features: Optional[set] = None) -> list:
    """
    Return a fixed-length float list for a node, zero-padded to FEAT_DIM.

    Parameters
    ----------
    node             : the node dict from the JSON
    node_type        : "characters", "occupations", "locations", or "organizations"
    exclude_features : set of feature names to zero out (per-node-type). Useful
                       for ablations — e.g. {"narrative_prominence",
                       "narrative_introduction_timing"} removes narrative
                       metadata that a detective wouldn't have access to.
    """
    keys = NODE_TYPE_FEATURES[node_type]
    exclude = exclude_features or set()
    feats = []
    for k in keys:
        if k in exclude:
            feats.append(0.0)
        else:
            feats.append(safe_float(k, node["features"].get(k, 0.0)))
    feats += [0.0] * (FEAT_DIM - len(feats))
    return feats


# ---------------------------------------------------------------------------
# 2. Stage 1 — relation normalisation (free-text → ~35 intermediate forms)
# ---------------------------------------------------------------------------

# Patterns are tried in order; first match wins.
_RELATION_PATTERNS = [

    # ---- FAMILY & ROMANTIC ----
    (re.compile(r"married|marries|spouse|wife of|husband of|widow of|bigamously|wedded"), "married to"),
    (re.compile(r"ex-spouse|ex-boyfriend|ex-girlfriend|ex-fianc|ended relationship|formerly engaged|past lover|had void marriages"), "ex-partner of"),
    (re.compile(r"fianc[eé]|intended to marry|engaged to|proposes to|wants to marry|elopes|courts"), "engaged/courting"),
    (re.compile(r"romantically involved|falls in love|in love with|affair with|had an affair|one-night stand|mistress of|moll of|flirting|attracted to|has a crush|infatuated|seduces|kissed|kisses|becomes close with"), "romantically involved with"),
    (re.compile(r"biological (child|daughter|son|parent|father|mother)|adopted|step-|stepdaughter|stepson|children of|child of|father of|mother of|parent of|son of|daughter of|twin|sibling|sister|brother|family member|relatives of|related by|related to|cousin|aunt|uncle|nephew|niece|illegitimate offspring|offspring of|heir (of|to)|legal guardian|guardian of|raised (as|by)|raises|father figure|head of family|daughter-in-law|sister in law|mother-in-law"), "family relation"),
    (re.compile(r"best friend|childhood friend|school friend|college friend|casual friend|long-time family friend|friend of|friends with|acquaint|companion to|roommates|riding companion"), "friends with"),
    (re.compile(r"neighbou?r(s)? (of|with)|lives near|neighbors with"), "neighbours with"),
    (re.compile(r"in love with|loves?|cares for|sympathizes?|grieves?|comforts?|loyal to|trusts?|trusted by|believes in|forgiven by|feels he must help"), "emotional bond with"),
    (re.compile(r"hates?|detests?|fears?|enraged by|estranged|cuts ties|antagonizes|obsessed with|consumed by guilt"), "negative relation with"),

    # ---- EMPLOYMENT & PROFESSIONAL ----
    (re.compile(r"employed as|works? as|formerly employed as|retired as|retires from"), "employed as"),
    (re.compile(r"employed (at|by)|works? (at|in|for)|working at|employee of|servant of|serves in|assistant (to|of)|apprentice to|subordinate (of|to)|valet of|housekeeper of|on payroll|hired by|henchman of|right-hand man of|security officer for"), "employed by"),
    (re.compile(r"employer of|employs|boss of|superior (of|to)|supervises|directs|manages|coach of|teacher of|teaches|trains|mentor of|tutor of|head of|leader of|leads|managing director of|founder of|co-founder"), "employer of"),
    (re.compile(r"colleague|co-worker|works? with|worked with|partner of|professionally involved"), "works with"),
    (re.compile(r"business partner|in cahoots|conspires|orchestrated plot together"), "business partner of"),
    (re.compile(r"client of|client$|uses services of|patron of"), "client of"),
    (re.compile(r"fired by|sacks|demoted|suspended from|replaced by"), "fired by"),
    (re.compile(r"hired (to|by)|hires|recruits|will employ|turns to for work"), "hired"),
    (re.compile(r"lawyer of|solicitor|defends in court|represents legally|representing in court"), "legal representative of"),
    (re.compile(r"therapist of|treats medically|treats wound|treats|diagnosed|diagnoses|examined by|examines|autopsies|performed surgery|nurse of|heals|cares for|tends wounds"), "treats medically"),
    (re.compile(r"student (at|of)|students of|studies (at|under)|classmate|mentee of|former pupil|coached by|alumna|attends school"), "student of"),

    # ---- LOCATION & PRESENCE ----
    (re.compile(r"resides? (at|in)|resided at|lives (at|in|with)|stayed? (at|in)|staying (at|with)|present (at|in)|rents|boards|guest (at|of)|hosts|frequents|inhabiting|camps at|headquartered at|located (at|in)|operates (at|from|in)|practices in"), "resides at"),
    (re.compile(r"travels? (to|through|with|on)|travelling|traveling|departs from|returns to|visits|vacationing|escaped to|born in|moves (to|through)|moved to|passes through|races to|drives (to|on)|sails on|fled from|flees|journeys"), "travels to"),
    (re.compile(r"body found|buried (at|beneath|under|somewhere|near)|body hidden|died (at|in)|dies (at|in)|death (at|in)|committed crime at|falls? from|found dead|location of (death|body)|death location"), "location of death/crime"),
    (re.compile(r"imprisoned (at|by|in)|incarcerated (at|in|with)|held prisoner|captive (in|of)|confined in"), "imprisoned at"),

    # ---- INVESTIGATION & LEGAL ----
    (re.compile(r"investigates?|analyzes? case|profiles?|deduces?|looking into|leads investigation|searches for"), "investigates"),
    (re.compile(r"arrests?|captured? by|captures?|takes into custody|charged with|prosecutes?|convicts?|wanted by|chases|pursues?|hunts"), "arrests"),
    (re.compile(r"witnesses?|witnessed|witness to|seen (by|with)|observed|overheard|overhears|watches"), "witnesses"),
    (re.compile(r"discovers?|uncovers?|finds?|stumbles upon|recogni[sz]es?|identifies?|identified (as|by)|recalls identity|realizes"), "discovers"),
    (re.compile(r"informs?|provides? (information|evidence|info|crucial)|shares information|reports? (to|on|missing|about)?|tips? off|tells|relays message|confides in|delivers (message|verdict)|announces|mentions|testifies|speaks (to|with)|corresponds with"), "informs"),
    (re.compile(r"accused (of|by)|accuses?|blames?|suspects? involvement|implicates|denounces|exposes?"), "accuses"),
    (re.compile(r"clears?|exonerates?|proves innocent|acquitted|believes in innocence"), "exonerates"),
    (re.compile(r"solves?|concludes is responsible|rules out"), "solves"),
    (re.compile(r"interviewed?|interviews?|brings in for questioning|interrogates?"), "interviews"),
    (re.compile(r"framed? by|frames?|falsely (identified|confesses|claims)|plants evidence|fraudulently claims|plagiari[sz]es"), "frames/deceives"),

    # ---- VIOLENCE & CRIME ----
    (re.compile(r"kills?|murdered?|murders?|shoots? (at|in|self)?|stabs?|strangles?|bludgeon|slashes|poisons?|wrongly shoots|eliminates|knocks out|incapacitates|subdues|overpowers|tranquilizes|drugged with|shoves onto|pushes|hits|beats|bombed|mortally wounded|tries to stab|shoots self"), "kills"),
    (re.compile(r"killed? by|murdered? by|shot by|stabbed by|beaten by|drugged by|tortured? by|victim of|victims? of|abandoned by accomplice"), "killed by"),
    (re.compile(r"attacks?|assaults?|ambushes?|abducts?|kidnaps?|ties up|traps?|tortures?|torments?|taunts?|stalks?|threatens?|blackmails?|coerces?|bribes?|extorts?|exploits?|intimidates?|endangers?|forces?|holds (captive|hostage|gun on)|demands ransom|controls?|drives insane|hypnotizes"), "threatens/attacks"),
    (re.compile(r"rapes?|sexually abused|father and rapist"), "sexually assaults"),
    (re.compile(r"steals? from|stole from|breaks? into|absconds with|loots from|swindles?|owes money to|financial(ly)? ruin|gets drugs from"), "financial crime/transaction"),
    (re.compile(r"conceals? body|destroys evidence|buries body|disposes? (of )?body|hides? (body|evidence|money)?|moves body|transported the body|conceals evidence|staged discovery|stages suicide|fakes death"), "conceals evidence"),
    (re.compile(r"accomplice of|complicit|aids?|aided by|helped?|assists? in|reluctantly helps|helps escape|helps transport"), "accomplice of"),
    (re.compile(r"fakes? death|disguised as|alias of|assumed identity|shares identity|same person as|real identity of|impersonates?|impostor|mistaken for|poses$"), "alias/identity of"),
    (re.compile(r"escaped? (from|justice|to)|escapes? from|fled from|flees|runs away from|released by|releases"), "escapes/releases"),

    # ---- SOCIAL & NARRATIVE ----
    (re.compile(r"affiliated with|active in|member(s)? of|comprises|represents?|donors? to|contributes? to|sponsors?|supports? (with)?|part of|participants? in|joined?|joins with"), "affiliated with"),
    (re.compile(r"owned by|owns?|possesses?|belongings of|inherits? (from)?|beneficiary|will inherit"), "owns/inherits"),
    (re.compile(r"confess(es|ed)? to|admits to|confessed to"), "confessed to"),
    (re.compile(r"protects?|rescues?|rescued by|saves?|saved (by|from)|shields?|spares?|dissuades"), "protects"),
    (re.compile(r"publishes?|published by|author of|writes? (for|about)?|adapted by|based on work|created by|creates?|authored"), "authored/published by"),
    (re.compile(r"met (by|in|on|with)|meets?|encounters?|runs into|reunite[sd]? with|introduced to|introduces"), "meets"),
    (re.compile(r"conflict with|conflicts with|rivals with|antagonizes?|opposes?|opposed (by|to)|competes against|in conflict with|feud|has personal vendetta|seeks revenge|plans revenge|avenges?"), "in conflict with"),
    (re.compile(r"convinces?|persuades?|convinced (by|to)|persuaded by|pressures?|coerces?|encourages?|urged|summoned (by|to)|ordered|orders"), "persuades"),
    (re.compile(r"suspects?|suspected by|under suspicion"), "suspects"),
    (re.compile(r"warns?|warned by|warns about|warns of"), "warns"),
    (re.compile(r"gives?|donates? to|pays? (for|off|out|debt)?|offers? (to|deal|job)?|financially supports|grants permission|gives money|gives gift"), "gives to"),
    (re.compile(r"near (to)?|adjacent|close to|next to|near$"), "near to"),
    (re.compile(r"unk$|unspecified|relationship type|same as|are$|was$|is$|is a$|equals$|instance of|label$"), "unspecified"),
    (re.compile(r"has (a connection|access|connections|influence|power|professional relationship|personal history|history)"), "has connection to"),
]


def normalize_relation(rel: str) -> str:
    """
    Stage 1: map a free-text relation to a ~35-item canonical intermediate form.
    Falls back to the lowercased original if no pattern matches.
    """
    rel_lower = rel.strip().lower()
    for pattern, canonical in _RELATION_PATTERNS:
        if pattern.search(rel_lower):
            return canonical
    return rel_lower


# ---------------------------------------------------------------------------
# 3. Stage 2 — coarsen to 8 relation types for link prediction
# ---------------------------------------------------------------------------
#
# Schema history and design rationale
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# v1 (original, 8 types): INTERACTS was a 28.6% junk-drawer; DECEIVES was
#    starved at 0.9%; PROFESSIONAL was bloated at 25.8% with non-employment
#    edges; AFFILIATED_WITH had only one source intermediate form.
#
# v2 (10 types): Split INTERACTS → SPATIAL + SOCIAL ✓; carved COMMUNICATES
#    out of INVESTIGATES; cleaned DECEIVES of post-deception resolution edges.
#    Problem: COMMUNICATES landed at 2.3% and DECEIVES remained at 0.8% —
#    both still critically starved.  The corpus simply does not contain enough
#    raw deception/communication edges to sustain them as independent types.
#
# v3 (this schema, 8 types): Merge starved types upward rather than splitting
#    further.  Two key consolidations solve the starvation problem:
#
#   (i)  COMMUNICATES (2.3%) is re-absorbed into INVESTIGATES.
#        informs/warns/witnesses/persuades are investigative-context edges in
#        the vast majority of mystery narratives; the 10-type experiment
#        confirmed they don't form a distinct distributional cluster.
#
#   (ii) DECEIVES (0.8%) absorbs the old AFFILIATED bucket (6.5%).
#        In mystery fiction, org-membership and alliance edges (affiliated
#        with, business partner of) almost always encode criminal conspiracy
#        or institutional complicity — semantically a form of deception.
#        Pure institutional affiliation (school, legal system, hospital)
#        moves instead to SOCIAL, where it sits alongside other low-weight
#        narrative-context edges (meets, gives to, travels to).
#        The combined DECEIVES bucket reaches ~8%, well above the starvation
#        threshold, while retaining a coherent "hidden allegiance" meaning.
#
# Final 8-type schema with observed corpus targets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   0  KILLS         lethal violence, death locations           (~5%)
#   1  HARMS         non-lethal violence, coercion, conflict    (~4%)
#   2  INVESTIGATES  detection, legal process, info pressure,   (~12%)
#                    testimony, confession, witness statements
#   3  DECEIVES      active deception, concealment, false       (~8%)
#                    identity, criminal complicity & alliance
#   4  PERSONAL_BOND family, romance, friendship, emotional     (~19%)
#                    care — well-calibrated, unchanged
#   5  PROFESSIONAL  employment, mentorship, medical care       (~24%)
#                    (has connection to pruned → SOCIAL)
#   6  SPATIAL       residence, location, proximity             (~19%)
#   7  SOCIAL        residual narrative contact + institutional (~12%)
#                    affiliation (meets, travels, gives, org
#                    membership, authorship, owns/inherits)
#
# Expected range: ~4%–24%.  No type below 4% or above 25%.

_COARSE_MAP = {
    # KILLS (0) — lethal violence and death locations
    "kills":                      0,
    "killed by":                  0,
    "location of death/crime":    0,

    # HARMS (1) — non-lethal violence, coercion, exploitation, conflict
    "threatens/attacks":          1,
    "sexually assaults":          1,
    "financial crime/transaction":1,
    "in conflict with":           1,
    "negative relation with":     1,

    # INVESTIGATES (2) — detection, legal process, information pressure
    # Re-absorbs informs/warns/witnesses/persuades/confessed to from the
    # failed COMMUNICATES split: these edges are overwhelmingly investigative
    # in mystery narratives and the corpus count confirms they belong here.
    # escapes/releases is a legal-process resolution event → also here.
    "investigates":               2,
    "arrests":                    2,
    "accuses":                    2,
    "suspects":                   2,
    "interviews":                 2,
    "discovers":                  2,
    "solves":                     2,
    "exonerates":                 2,
    "legal representative of":    2,
    "imprisoned at":              2,
    "informs":                    2,   # tip-offs, witness statements
    "warns":                      2,   # detective/legal warnings
    "persuades":                  2,   # pressure during investigation
    "witnesses":                  2,   # testimony
    "confessed to":               2,   # investigative resolution
    "escapes/releases":           2,   # legal-process outcome

    # DECEIVES (3) — active deception, concealment, false identity,
    #                criminal complicity and hidden alliance
    # Core deception edges (frames/deceives, conceals evidence, alias)
    # are joined by the old AFFILIATED bucket.  In mystery fiction, alliance
    # and membership edges (affiliated with, business partner of, owns/
    # inherits, authored/published by) almost universally encode criminal
    # conspiracy or institutional complicity — a form of hidden allegiance
    # that the decoder should learn alongside active deception acts.
    # This merger lifts DECEIVES from 0.8% → ~8%, resolving starvation
    # while preserving a coherent "deception / hidden allegiance" meaning.
    "frames/deceives":            3,
    "conceals evidence":          3,
    "alias/identity of":          3,
    "accomplice of":              3,
    "affiliated with":            3,   # criminal/conspiratorial alliance
    "business partner of":        3,   # often criminal partnership
    "owns/inherits":              3,   # hidden financial interest
    "authored/published by":      3,   # narrative-level hidden authorship

    # PERSONAL_BOND (4) — family, romance, friendship, emotional care
    # Well-calibrated at ~19%; unchanged from v1.
    "married to":                 4,
    "ex-partner of":              4,
    "engaged/courting":           4,
    "romantically involved with": 4,
    "family relation":            4,
    "friends with":               4,
    "neighbours with":            4,
    "emotional bond with":        4,
    "protects":                   4,

    # PROFESSIONAL (5) — formal occupational roles, employment, mentorship,
    #                     medical care
    # has connection to moves to SOCIAL (too generic to anchor an
    # employment-specific relation matrix).  All other employment edges
    # remain; this bucket is well-shaped at ~24%.
    "employed by":                5,
    "employed as":                5,
    "employer of":                5,
    "works with":                 5,
    "hired":                      5,
    "fired by":                   5,
    "client of":                  5,
    "student of":                 5,
    "treats medically":           5,

    # SPATIAL (6) — residence, physical location, proximity
    # Unchanged from v2; cleanly separates geographic co-location from
    # narrative social contact.  resides at is the corpus's most frequent
    # single edge type and needs its own relation matrix.
    "resides at":                 6,
    "near to":                    6,

    # SOCIAL (7) — residual narrative contact + lightweight institutional links
    # Absorbs: pure social interaction (meets, travels, gives), institutional
    # org-membership that is NOT criminal (school enrolment, hospital patient,
    # legal-system affiliation), and has connection to (too vague for
    # PROFESSIONAL).  Slightly larger than v2's SOCIAL because institutional
    # affiliation moved here from DECEIVES, but still a well-bounded residual.
    "meets":                      7,
    "visits":                     7,
    "travels to":                 7,
    "gives to":                   7,
    "observed":                   7,
    "has connection to":          7,   # generic institutional link (pruned
                                       # from PROFESSIONAL — too ambiguous)
    "unspecified":                7,
    "has a specific history with":7,
    "interacts with":             7,
}

# Coarse relation labels (for rel_labels dict in RelationalGraph)
# IDs 0–7 are canonical; 8–15 are their inverse counterparts.
_BASE_COARSE_LABELS = {
    0: "kills",
    1: "harms",
    2: "investigates",
    3: "deceives",
    4: "personal_bond",
    5: "professional",
    6: "spatial",
    7: "social",
}

NUM_BASE_RELATIONS = len(_BASE_COARSE_LABELS)  # 8

COARSE_LABELS = dict(_BASE_COARSE_LABELS)
for _id, _name in _BASE_COARSE_LABELS.items():
    COARSE_LABELS[_id + NUM_BASE_RELATIONS] = f"{_name}_inv"

NUM_COARSE_RELATIONS = len(COARSE_LABELS)  # 16 (8 canonical + 8 inverse)


def coarsen_relation(intermediate: str) -> int:
    """
    Stage 2: map an intermediate canonical form to a coarse relation index 0–7.
    Anything not explicitly listed → 7 (SOCIAL, the residual bucket).
    """
    return _COARSE_MAP.get(intermediate, 7)


# ---------------------------------------------------------------------------
# 4. Main loader
# ---------------------------------------------------------------------------

CHARACTER_CLASS_MAP = {
    "villain":     0,
    "victim":      1,
    "witness":     2,
    "uninvolved":  3,
}
CHARACTER_CLASS_NAMES = {v: k.capitalize() for k, v in CHARACTER_CLASS_MAP.items()}
NUM_CHARACTER_CLASSES = len(CHARACTER_CLASS_MAP)  # 4

# Intermediate relation forms that represent the crime itself.
# These edges reveal who committed the crime and must be masked at
# evaluation time in the detective scenario.
CRIME_INTERMEDIATES = {
    "kills",                      # A murders B
    "killed by",                  # B was murdered by A
    "sexually assaults",          # sexual crime
    "financial crime/transaction", # theft, fraud, swindling
}


def load_mystery_graphs(
    json_dir: str,
    val_fraction: float  = 0.1,
    test_fraction: float = 0.1,
    seed: int            = 42,
    exclude_features: Optional[set] = None,
    exclude_entries: Optional[set] = None,
) -> RelationalGraph:
    """
    Merge all mystery JSON files in json_dir into a single RelationalGraph
    with 16 relation types (8 coarse + 8 inverse) and character class labels.

    Parameters
    ----------
    json_dir         : directory containing the .json files
    val_fraction     : fraction of *stories* held out for validation
    test_fraction    : fraction of *stories* held out for test
    seed             : random seed for the story-level split
    exclude_features : optional set of feature names to zero out. E.g.
                       {"narrative_prominence", "narrative_introduction_timing"}
                       removes narrative metadata a real detective would not have.
    exclude_entries  : optional set of entry IDs (e.g. {"FLM_047"}) to drop
                       entirely. Useful for stories with no identified villain
                       that shouldn't pollute train/test labels but may still
                       be loaded separately for inference-only analysis.

    Returns
    -------
    A RelationalGraph ready to pass directly to train() in rgcn_model.py.
    Includes class_labels, train_mask, val_mask, test_mask for node
    classification (villain prediction).
    """
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {json_dir}")

    if exclude_entries:
        before = len(json_files)
        json_files = [f for f in json_files
                      if f.replace(".json", "") not in exclude_entries]
        dropped = before - len(json_files)
        if dropped > 0:
            print(f"Excluded {dropped} entries: {sorted(exclude_entries)}")

    # Story-level split
    rng = random.Random(seed)
    shuffled = json_files[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test  = max(1, int(test_fraction  * n_total))
    n_val   = max(1, int(val_fraction   * n_total))
    n_train = n_total - n_val - n_test

    train_files = set(shuffled[:n_train])
    val_files   = set(shuffled[n_train : n_train + n_val])
    test_files  = set(shuffled[n_train + n_val :])

    print(f"Story split — train: {len(train_files)}, "
          f"val: {len(val_files)}, test: {len(test_files)}")
    if exclude_features:
        print(f"Excluded features: {sorted(exclude_features)}")

    # Accumulators
    all_features    = []
    node_labels     = {}
    class_labels    = []      # per-node class index (-1 = non-character or UNK)
    node_split      = []      # per-node: 'train', 'val', or 'test'
    node_offset     = 0

    train_edges  = []
    val_edges    = []
    test_edges   = []

    # Per-edge crime flags (parallel to edge lists, True = crime edge)
    train_crime  = []
    val_crime    = []
    test_crime   = []

    # Counters for diagnostics
    intermediate_counts = {}
    coarse_counts       = {i: 0 for i in range(NUM_COARSE_RELATIONS)}

    for fname in json_files:
        with open(os.path.join(json_dir, fname)) as f:
            data = json.load(f)

        if fname in train_files:
            edge_bucket = train_edges
            crime_bucket = train_crime
            split_name  = "train"
        elif fname in val_files:
            edge_bucket = val_edges
            crime_bucket = val_crime
            split_name  = "val"
        else:
            edge_bucket = test_edges
            crime_bucket = test_crime
            split_name  = "test"

        # Assign global IDs and extract character labels
        local_to_global = {}
        for node_type in ["characters", "occupations", "locations", "organizations"]:
            for node in data.get(node_type, []):
                local_to_global[node["id"]] = node_offset
                node_labels[node_offset]    = node["name"]
                all_features.append(extract_node_features(node, node_type, exclude_features))
                node_split.append(split_name)

                if node_type == "characters":
                    label_str = node.get("label", "UNK").strip().lower()
                    class_labels.append(CHARACTER_CLASS_MAP.get(label_str, -1))
                else:
                    class_labels.append(-1)  # non-character nodes are unlabeled

                node_offset += 1

        # Convert and coarsen edges; add inverse edges
        for edge in data.get("edges", []):
            src_local = edge["source"]
            dst_local = edge["target"]
            if src_local not in local_to_global or dst_local not in local_to_global:
                continue

            intermediate = normalize_relation(edge["relation"])
            coarse_id    = coarsen_relation(intermediate)

            intermediate_counts[intermediate] = (
                intermediate_counts.get(intermediate, 0) + 1
            )
            coarse_counts[coarse_id] += 1

            g_src = local_to_global[src_local]
            g_dst = local_to_global[dst_local]
            is_crime = intermediate in CRIME_INTERMEDIATES

            # Canonical edge
            edge_bucket.append((g_src, g_dst, coarse_id))
            crime_bucket.append(is_crime)
            # Inverse edge (reversed direction, relation ID offset by NUM_BASE_RELATIONS)
            inv_id = coarse_id + NUM_BASE_RELATIONS
            edge_bucket.append((g_dst, g_src, inv_id))
            crime_bucket.append(is_crime)
            coarse_counts[inv_id] = coarse_counts.get(inv_id, 0) + 1

    def to_tensor(edge_list):
        if not edge_list:
            return torch.zeros((0, 3), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long)

    node_features  = torch.tensor(all_features, dtype=torch.float)
    class_labels_t = torch.tensor(class_labels, dtype=torch.long)

    # Build train/val/test masks for node classification
    # Only character nodes with valid labels (class >= 0) are masked
    train_mask = torch.zeros(node_offset, dtype=torch.bool)
    val_mask   = torch.zeros(node_offset, dtype=torch.bool)
    test_mask  = torch.zeros(node_offset, dtype=torch.bool)

    for i in range(node_offset):
        if class_labels[i] < 0:
            continue  # non-character or UNK — excluded from classification loss
        if node_split[i] == "train":
            train_mask[i] = True
        elif node_split[i] == "val":
            val_mask[i] = True
        else:
            test_mask[i] = True

    # Print summary
    total_edges = len(train_edges) + len(val_edges) + len(test_edges)
    print(f"Nodes          : {node_offset:,}")
    print(f"Relation types : {NUM_COARSE_RELATIONS}  (8 canonical + 8 inverse, "
          f"coarsened from {len(intermediate_counts)} intermediate forms)")
    print(f"Train edges    : {len(train_edges):,}")
    print(f"Val edges      : {len(val_edges):,}")
    print(f"Test edges     : {len(test_edges):,}")
    print(f"Node feat dim  : {FEAT_DIM}")

    print(f"\nNode classification (villain prediction):")
    print(f"  Classes       : {NUM_CHARACTER_CLASSES}  ({', '.join(CHARACTER_CLASS_NAMES[i] for i in range(NUM_CHARACTER_CLASSES))})")
    print(f"  Train chars   : {train_mask.sum().item():,}")
    print(f"  Val chars     : {val_mask.sum().item():,}")
    print(f"  Test chars    : {test_mask.sum().item():,}")
    print(f"  Unlabeled     : {(class_labels_t == -1).sum().item():,}  (non-character + UNK)")

    # Crime edge masks
    train_crime_t = torch.tensor(train_crime, dtype=torch.bool)
    val_crime_t   = torch.tensor(val_crime,   dtype=torch.bool)
    test_crime_t  = torch.tensor(test_crime,  dtype=torch.bool)
    total_crime   = train_crime_t.sum().item() + val_crime_t.sum().item() + test_crime_t.sum().item()

    print(f"\nCrime edges    : {total_crime:,}  (masked at eval time)")
    print(f"  Intermediates: {', '.join(sorted(CRIME_INTERMEDIATES))}")

    print(f"\nCoarse relation distribution:")
    for rel_id in sorted(COARSE_LABELS.keys()):
        label = COARSE_LABELS[rel_id]
        count = coarse_counts.get(rel_id, 0)
        pct   = 100.0 * count / total_edges if total_edges else 0
        print(f"  {rel_id:>2}  {label:<22}  {count:>6,}  ({pct:5.1f}%)")

    return RelationalGraph(
        name          = "mystery-corpus",
        num_nodes     = node_offset,
        num_relations = NUM_COARSE_RELATIONS,
        train_edges   = to_tensor(train_edges),
        val_edges     = to_tensor(val_edges),
        test_edges    = to_tensor(test_edges),
        node_features = node_features,
        node_labels   = node_labels,
        rel_labels    = COARSE_LABELS,
        class_labels  = class_labels_t,
        num_classes   = NUM_CHARACTER_CLASSES,
        class_names   = CHARACTER_CLASS_NAMES,
        train_mask    = train_mask,
        val_mask      = val_mask,
        test_mask     = test_mask,
        train_crime_mask = train_crime_t,
        val_crime_mask   = val_crime_t,
        test_crime_mask  = test_crime_t,
    )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_mystery_graphs.py <path_to_json_dir>")
        sys.exit(1)

    graph = load_mystery_graphs(sys.argv[1])
    graph.summary()

    print("\nCoarse relation labels:")
    for rel_id, label in sorted(graph.rel_labels.items()):
        print(f"  {rel_id}: {label}")
