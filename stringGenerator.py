
def getFormattedStruSentence(index):
    if index == 1 or index == 2:
        s = "[MASK]{Mask_Suffix} {PlaceHolder} %s{Trait_Suffix}"
    elif index == 3:
        s = "%s{Trait_Suffix} [MASK]{Mask_Suffix} {PlaceHolder}"

    return s

def processTraitSuffix(trait_suffix):
    # print(trait_suffix)
    trait_suffix = trait_suffix.strip()
    if len(trait_suffix) == 0:
        return ""
    
    if trait_suffix[0] == '-':
        trait_suffix = trait_suffix[1:]
    else :
        trait_suffix = " "+trait_suffix
    return trait_suffix