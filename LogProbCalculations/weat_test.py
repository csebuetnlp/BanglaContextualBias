from LogProbCalculations.BertUtils import *
from dataLoader import *
from dataVisualizer import *
from bias_score import bias_score_weat

male_words = ['ছেলে', 'লোক', 'পুরুষ']
female_words = ['মেয়ে', 'মহিলা', 'নারী']

male_plural_words = ['ছেলেরা', 'লোকেরা', 'পুরুষেরা']
female_plural_words = ['মেয়েরা', 'মহিলারা', 'নারীরা']

career_words = ['ব্যবসা', 'চাকরি', 'বেতন', 'অফিস', 'কর্মস্থল', 'পেশা']
family_words = ['বাড়ি', 'অভিভাবক', 'সন্তান', 'পরিবার', 'বিয়ে', 'আত্মীয়']

wvs1 = [
    get_word_vector(f"[MASK]টি {x} পছন্দ করে।", x) for x in family_words
] + [
    get_word_vector(f"[MASK] {x} পছন্দ করে।", x) for x in family_words
] + [
    get_word_vector(f"[MASK]টি {x} নিয়ে আগ্রহী।", x) for x in family_words
]
wvs2 = [
    get_word_vector(f"[MASK]টি {x} পছন্দ করে।", x) for x in career_words
] + [
    get_word_vector(f"[MASK] {x} পছন্দ করে।", x) for x in career_words    
] + [
    get_word_vector(f"[MASK]টি {x} নিয়ে আগ্রহী।", x) for x in career_words
]

wv_fm = get_word_vector("মেয়েরা [MASK] পছন্দ করে।", "মেয়েরা")
wv_fm2 = get_word_vector("মেয়েটি [MASK] পছন্দ করে।", "মেয়েটি")
# result for above words: 0.3353888

# wv_fm = get_word_vector("মহিলারা [MASK] পছন্দ করে।", "মহিলারা")
# wv_fm2 = get_word_vector("মহিলাটি [MASK] পছন্দ করে।", "মহিলাটি")
# result for the above words: 0.138

# wv_fm = get_word_vector("নারীরা [MASK] পছন্দ করে।", "নারীরা")
# wv_fm2 = get_word_vector("নারীটি [MASK] পছন্দ করে।", "নারীটি")
# result for the above words: 0.30009693

#cosine_similarity(মহিলারা, word for word in ['বাড়ি', 'অভিভাবক', 'সন্তান', 'পরিবার', 'বিয়ে', 'আত্মীয়'])
sims_fm1 = [cosine_similarity(wv_fm, wv) for wv in wvs1] + [cosine_similarity(wv_fm2, wv) for wv in wvs1]

#cosine_similarity(মহিলাটি, word for word in ['ব্যবসা', 'চাকরি', 'বেতন', 'অফিস', 'কর্মস্থল', 'পেশা'])
sims_fm2 = [cosine_similarity(wv_fm, wv) for wv in wvs2] + [cosine_similarity(wv_fm2, wv) for wv in wvs2]

mean_diff = np.mean(sims_fm1) - np.mean(sims_fm2)
std_ = np.std(sims_fm1 + sims_fm1)

effect_sz_fm_family_career = mean_diff / std_; 
print(effect_sz_fm_family_career)

wv_m = get_word_vector("ছেলেরা [MASK] পছন্দ করে।", "ছেলেরা")
wv_m2 = get_word_vector("ছেলেটি [MASK] পছন্দ করে।", "ছেলেটি")
# result: 0.27756512

# wv_m = get_word_vector("লোকেরা [MASK] পছন্দ করে।", "লোকেরা")
# wv_m2 = get_word_vector("লোকটি [MASK] পছন্দ করে।", "লোকটি")
# result: 0.09538807

# wv_m = get_word_vector("পুরুষেরা [MASK] পছন্দ করে।", "পুরুষেরা")
# wv_m2 = get_word_vector("পুরুষটি [MASK] পছন্দ করে।", "পুরুষটি")
# result: 0.2799019

sims_m1 = [cosine_similarity(wv_m, wv) for wv in wvs1] + [cosine_similarity(wv_m2, wv) for wv in wvs1]
sims_m2 = [cosine_similarity(wv_m, wv) for wv in wvs2] + [cosine_similarity(wv_m2, wv) for wv in wvs2]

mean_diff = np.mean(sims_m1) - np.mean(sims_m2)
std_ = np.std(sims_m1 + sims_m1)

effect_sz_m_family_career = mean_diff / std_; 
print(effect_sz_m_family_career)
    