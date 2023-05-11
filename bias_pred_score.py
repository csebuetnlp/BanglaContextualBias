from BertUtils import *
from dataLoader import *
from dataVisualizer import *
from collections import defaultdict
import pandas as pd

def convertToDictOfLists(container):
    '''
        This function converts a list of dictionaries to a dictionary of lists
    '''
    dict_of_lists = defaultdict(list)
    for d in container:
        for k, v in d.items():
            if k in dict_of_lists.keys():
                dict_of_lists[k].append(v)
            else:
                dict_of_lists[k] = [v] 
    return dict_of_lists

def bias_score_aux(sentence, trait, gender_words, use_last_mask, apply_softmax=False, calc_tgt_fill_prob=False):
    subject_fill_logits = get_mask_fill_logits(
        sentence.replace("XXX", trait).replace("GGG", "[MASK]"), 
        gender_words, apply_softmax=apply_softmax
    )
    subject_fill_prior_logits = get_mask_fill_logits(
        sentence.replace("XXX", "[MASK]").replace("GGG", "[MASK]"), 
        gender_words, use_last_mask=use_last_mask, apply_softmax=apply_softmax
    )

    norm_logits = {gw: subject_fill_logits[gw]-subject_fill_prior_logits[gw] for gw in gender_words}

    tgt_fill_prob = {}

    if calc_tgt_fill_prob:
    
        for gw in gender_words:
            tgt_fill_prob[gw] = get_mask_fill_logits(
                sentence.replace("GGG", gw).replace("XXX", "[MASK]"), [trait],
                apply_softmax=True,
            )[trait]

    return norm_logits, tgt_fill_prob


def getFillScore(df, gender_words, use_last_mask):
    fill_logits_container = []
    fill_preds_container = []
    #iterate over the dataframe rows
    for index, row in df.iterrows():
        trait = row['Trait']
        mask_sentence = row['Mask_Sent']
        fill_logits = get_mask_fill_logits(mask_sentence%(trait), gender_words, use_last_mask=use_last_mask)
        fill_preds = get_mask_fill_logits(mask_sentence%(trait), gender_words, apply_softmax=True, use_last_mask=use_last_mask)

        fill_logits_container.append(fill_logits)
        fill_preds_container.append(fill_preds)

    return convertToDictOfLists(fill_logits_container), convertToDictOfLists(fill_preds_container)


def getBiasNormScore(df, gender_words, use_last_mask):
    norm_logits_container = []
    norm_preds_container = []
    target_fill_prob_container = []
    #iterate over the dataframe rows
    for index, row in df.iterrows():
        trait = row['Trait']
        bias_sentence = row['Bias_Sent']
        norm_logits, _ = bias_score_aux(bias_sentence, trait, gender_words, use_last_mask = use_last_mask)
        norm_preds, target_fill_prob = bias_score_aux(bias_sentence, trait, gender_words, 
                                                      use_last_mask, apply_softmax=True, calc_tgt_fill_prob=True)
        norm_logits_container.append(norm_logits)
        norm_preds_container.append(norm_preds)
        target_fill_prob_container.append(target_fill_prob)
    return convertToDictOfLists(norm_logits_container), \
            convertToDictOfLists(norm_preds_container), \
            convertToDictOfLists(target_fill_prob_container)

def calculateLogitScores(df, title, gendered_words, use_last_mask=False):
    # iterate for gendered words
    bias_score_dict = dict()
    bias_score_dict["Attribute"] = df["Trait"].tolist()

    gender_fill_logits, gender_fill_preds = getFillScore(df, gendered_words, use_last_mask)
    norm_logits, norm_preds, target_fill_prob = getBiasNormScore(df, gendered_words, use_last_mask)
    
    for i in range(0, len(gendered_words), 2):
        mw = gendered_words[i]
        fw = gendered_words[i+1]
        bias_score_dict["logits-"+mw] = gender_fill_logits[mw]
        bias_score_dict["logits-"+fw] = gender_fill_logits[fw]

        bias_score_dict["normlogits-"+mw] = norm_logits[mw]
        bias_score_dict["normlogits-"+fw] = norm_logits[fw]

        bias_score_dict["preds-"+mw] = gender_fill_preds[mw]
        bias_score_dict["preds-"+fw] = gender_fill_preds[fw]

        bias_score_dict["normpreds-"+mw] = norm_preds[mw]
        bias_score_dict["normpreds-"+fw] = norm_preds[fw]

        bias_score_dict["tgtfillprob-"+mw] = target_fill_prob[mw]
        bias_score_dict["tgtfillprob-"+fw] = target_fill_prob[fw]

    score_df = pd.DataFrame(bias_score_dict)
    score_df.to_csv("./results_new/"+title+"_scores.csv", index=False)



if __name__ == '__main__':
    csv_df_groups = loadAllCSVfromFolder("./data")
    gendered_words_list = getGenderedWords()
    gendered_words = []

    for lst in gendered_words_list:
        gendered_words.extend(lst)

    all_average_container = []
    for group in csv_df_groups:
        group_title = group["title"]
        elements = group["group"]

        avg_score_for_group = []
        comparison_list_container = []

        for element in elements:
            element_title = element["title"]
            df = element["df"]
            use_last_mask = element["use_last_mask"]
            print("Processing: ", element_title)
            comparison_list, gender_avg_score = calculateLogitScores(df, element_title, gendered_words=gendered_words, use_last_mask=use_last_mask)
            