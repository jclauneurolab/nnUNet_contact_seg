map_acronyms = {
    "A": "Anterior",
    "L": "Lateral",
    "C": "Center",
    "Ce": "Central",
    "P": "Posterior",
    "M": "Medial",
    "I": "Interior",
    "S": "Superior",
    "Fr": "Frontal",
    "Inf": "Infra",
    "Int": "Intra",
    "Ln": "Line",
    "Ma": "Margin",
    "Me": "Medial",
    "Mi": "Middle",
    "Mo": "Motor",
    "Or": "Orthogonal",
    "Parc": "Paracentral",
    "Par": "Para",
    "Pr": "Pre",
    "Sa": "Sagittal",
    "Sp": "Supplementary",
    "Sup": "Supra",
    "Sc": "Sensory",
    "Mc": "Motor cortex",
    "Ps": "Parasagittal",
    "Am": "Amygdala",
    "An": "Angular",
    "Ca": "Calcarine",
    "Br": "Broca",
    "Cc": "Corpus Callosum",
    "Cg": "Cingulate",
    "Cl": "Cleft",
    "Cu": "Cuneus",
    "Cx": "Convex",
    "Dy": "Dysplasia",
    "Ec": "Entorhinal Cortex",
    "Fa": "Face",
    "Gy": "Gyrus",
    "Ha": "Hand",
    "Hc": "Hippocampus",
    "Hs": "Heschel’s",
    "Ht": "heterotopia",
    "In": "Insula",
    "Lg": "Lingula",
    "Lo": "Lobe",
    "Ls": "Lesion",
    "O": "Orbital",
    "Oc": "Occipital",
    "Op": "Operculum",
    "Po": "Pole",
    "Pa": "Parietal",
    "Ps": "Parasagittal",
    "Ro": "Rolandic",
    "Re": "Resection",
    "Rs": "Rostrum",
    "Sl": "Splenium",
    "SMA": "Supplementary Motor Area",
    "Te": "Temporal",
    "Unc": "Uncus",
    "We": "Wernicke",
    "Mes": "Mesial",
    "MFG": "Middle Frontal Gyrus"
}

map_hemi = {
    "L": "Left",
    "R": "Right"
}

def convert_acronym_to_words(label):
    label = label.strip()

    # save original label in case label is not matched
    original_label = label 
    words = []
    
    # add logic for left and right 
    if label and label[0] in map_hemi:
        words.append(map_hemi[label[0]])
        label = label[1:]
    i = 0

    while i < len(label):
        matched = False
        
        # greedy approach due to "SMA" being out of pattern
        # looks at a combination of 3 letters first, then 2, then 1
        for j in range(3, 0, -1):
            part = label[i:i+j]
            if part in map_acronyms:
                words.append(map_acronyms[part])
                i += j
                matched = True
                break
        if not matched:
            return original_label
    return " ".join(words)




