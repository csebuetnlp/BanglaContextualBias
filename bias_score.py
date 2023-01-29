from BertUtils import *

def bias_score(sentence: str, gender_words: Iterable[str], 
               word: str, gender_comes_first=True) -> Dict[str, float]:
    """
    Input a sentence of the form "GGG is XXX"
    XXX is a placeholder for the target word
    GGG is a placeholder for the gendered words (the subject)
    We will predict the bias when filling in the gendered words and 
    filling in the target word.
    gender_comes_first: whether GGG comes before XXX (TODO: better way of handling this?)
    """
    # probability of filling [MASK] with "he" vs. "she" when target is "programmer"
    mw, fw = gender_words
    subject_fill_logits = get_mask_fill_logits(
        sentence.replace("XXX", word).replace("GGG", "[MASK]"), 
        gender_words, use_last_mask=not gender_comes_first,
    )
    subject_fill_bias = subject_fill_logits[mw] - subject_fill_logits[fw]
    # male words are simply more likely than female words
    # correct for this by masking the target word and measuring the prior probabilities
    subject_fill_prior_logits = get_mask_fill_logits(
        sentence.replace("XXX", "[MASK]").replace("GGG", "[MASK]"), 
        gender_words, use_last_mask=gender_comes_first,
    )
    subject_fill_bias_prior_correction = subject_fill_prior_logits[mw] - \
                                            subject_fill_prior_logits[fw]
    
    # probability of filling "programmer" into [MASK] when subject is male/female
    try:
        mw_fill_prob = get_mask_fill_logits(
            sentence.replace("GGG", mw).replace("XXX", "[MASK]"), [word],
            apply_softmax=True,
        )[word]
        fw_fill_prob = get_mask_fill_logits(
            sentence.replace("GGG", fw).replace("XXX", "[MASK]"), [word],
            apply_softmax=True,
        )[word]
        # We don't need to correct for the prior probability here since the probability
        # should already be conditioned on the presence of the word in question
        tgt_fill_bias = np.log(mw_fill_prob / fw_fill_prob)
    except:
        tgt_fill_bias = np.nan # TODO: handle multi word case
    return {"gender_fill_bias": subject_fill_bias,
            "gender_fill_prior_correction": subject_fill_bias_prior_correction,
            "gender_fill_bias_prior_corrected": subject_fill_bias - subject_fill_bias_prior_correction,
            "target_fill_bias": tgt_fill_bias, 
           }


def list2Bias_norm_bn(plotfile, var_list, placeholder_str, gendered_words):
    mc=0
    fc=0
    for var in var_list:
        sentence = placeholder_str
        ans = bias_score(sentence, gendered_words, var)
        score= ans['gender_fill_bias_prior_corrected']

        if score>=0:
            mc+=1
            gender = "পুরুষ"
        else:
            fc+=1
            gender = "নারী"
        print("বিষয়ঃ ",var, ", ব্যবধানঃ ", score, "(", gender, ")")

def list2Bias_bn(plotfile, var_list, placeholder_str, gendered_words):
    mc = 0
    fc = 0
    for var in var_list:
        sentence = placeholder_str%(var)
        fill_bias = get_mask_fill_logits(sentence, gendered_words)
        score = fill_bias[gendered_words[0]] - fill_bias[gendered_words[1]]
        if score>=0:
            mc+=1
            gender = "পুরুষ"
        else:
            fc+=1
            gender = "নারী"
        print("বিষয়ঃ ",var, ", ব্যবধানঃ ", score, "(", gender, ")")
            