from BertUtils import *
from dataLoader import *
from dataVisualizer import *

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
            "female_fill_bias_prior": subject_fill_prior_logits[fw]
           }


def getFillScore(df, gender_words):
    mc = 0
    fc = 0
    score = []
    female_fill_bias = []
    #iterate over the dataframe rows
    for index, row in df.iterrows():
        trait = row['Trait']
        mask_sentence = row['Mask_Sent']
        fill_bias = get_mask_fill_logits(mask_sentence%(trait), gender_words)
        fill_score = fill_bias[gender_words[0]] - fill_bias[gender_words[1]] 
        score.append(fill_score)
        if fill_score>=0:
            mc+=1
        else:
            fc+=1
        female_fill_bias.append(fill_bias[gender_words[1]])
    return score, mc, fc, female_fill_bias


def getBiasNormScore(df, gender_words):
    mc = 0
    fc= 0
    score = []
    female_prior_bias = []
    #iterate over the dataframe rows
    for index, row in df.iterrows():
        trait = row['Trait']
        bais_sentence = row['Bias_Sent']
        ans = bias_score(bais_sentence, gender_words, trait)
        prior_score= ans['gender_fill_bias_prior_corrected']
        score.append(prior_score)
        if prior_score>=0:
            mc+=1
        else:
            fc+=1
        female_prior_bias.append(ans["female_fill_bias_prior"])
    return score, mc, fc, female_prior_bias


def calculateDfAverage(bias_score_list):
    fill_scores = []
    norm_scores = []
    for gender_word_scores in bias_score_list:
        fill_scores.append(gender_word_scores["fill_score"])
        norm_scores.append(gender_word_scores["norm_score"])

    fill_scores_avg = [sum(elements)/len(elements) for elements in zip(*fill_scores)] 
    norm_scores_avg = [sum(elements)/len(elements) for elements in zip(*norm_scores)]
    mc_f = len([x for x in fill_scores_avg if x>=0])
    fc_f = len(fill_scores_avg) - mc_f
    mc_norm = len([x for x in norm_scores_avg if x>=0])
    fc_norm = len(norm_scores_avg) - mc_norm

    return {"fill_score_avg": fill_scores_avg, 
            "norm_score_avg": norm_scores_avg,
            "mc_f": mc_f,
            "fc_f": fc_f,
            "mc_norm": mc_norm,
            "fc_norm": fc_norm} 


def calculateScore(df, title, gendered_words=[["ছেলে", "মেয়ে"], ["পুরুষ", "নারী"]]):
    # iterate for gendered words
    bias_score_dict = dict()
    bias_score_dict["Attribute"] = df["Trait"].tolist()
    bias_score_list = []
    comparison_list = []
    for gender_word in gendered_words:
        fill_scores, mc_f, fc_f, _ = getFillScore(df, gender_words=gender_word)
        norm_score, mc_norm, fc_norm, _ = getBiasNormScore(df, gender_words=gender_word)
        bias_score_list.append(
            {
                "fill_score": fill_scores,
                "norm_score": norm_score,
            }
        )
        comparison_list.append(
            {
                "gender": gender_word,
                "mc_f": mc_f,
                "fc_f": fc_f,
                "mc_norm": mc_norm,
                "fc_norm": fc_norm
            }
        )
        bias_score_dict["Bias_Score("+gender_word[0]+"-"+gender_word[1]+")"] = fill_scores
        bias_score_dict["Norm_Score("+gender_word[0]+"-"+gender_word[1]+")"] = norm_score
    
    avg_score = calculateDfAverage(bias_score_list)

    bias_score_dict["Bias_Score(Avg)"] = avg_score["fill_score_avg"]
    bias_score_dict["Norm_Score(Avg)"] = avg_score["norm_score_avg"]
    score_df = pd.DataFrame(bias_score_dict)
    score_df.to_csv("./results/"+title+"_scores.csv", index=False)
    return comparison_list, avg_score


if __name__ == '__main__':
    csv_df_list = loadAllCSVfromFolder("./data")
    avg_scores_for_title = []
    for csv_df in csv_df_list:
        title = csv_df["title"]
        df = csv_df["df"]
        print("Processing: ", title)
        comparison_list, avg_score = calculateScore(df, title)
        avg_scores_for_title.append(
            {
                "title": title,
                "avg_": avg_score,
            }
        )
        processed_data = formatProcessorForGenderedWords(comparison_list, title)
        createSubplotForPiePlot(processed_data)

    # Create pieplot for avg data
    avg_processed_data = formatProcessorForAvgScore(avg_scores_for_title, "Average For All Traits")
    createSubplotForPiePlot(avg_processed_data)


        