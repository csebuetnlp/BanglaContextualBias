
'''
This file contains the functions that are used to visualize the data.
The format for data argument in createSubplotForPiePlot() is:
    {
        "title": "title",
        "data": [
            {
                "labels": [gendered words],
                "values": [values- mc and fc],
                "norm_values": [values- mc_norm and fc_norm],
            } ...
        ]
    } 

'''

import matplotlib.pyplot as plt
import math

def createSubplotForPiePlot(data, save=True):
    title = data["title"]
    data = data["data"]
    fig, axs = plt.subplots(math.ceil((len(data)*2)/4), 4, figsize=(20, 40))
    fig.suptitle(title)
    for i, d in enumerate(data):
        labels = d["labels"][0] + "-" + d["labels"][1]
        values = d["values"]
        norm_values = d["norm_values"]
        axs[(2*i)//4, (2*i)%4].pie(values, labels=labels, autopct='%.2f%%')
        axs[(2*i)//4, (2*i)%4].set_title("Fill Score")
        axs[(2*i + 1)//4, (2*i + 1)%4].pie(norm_values, labels=labels, autopct='%.2f%%')
        axs[(2*i + 1)//4, (2*i + 1)%4].set_title("Norm Score")

    if save:
        plt.savefig("./results/"+title+".png")

def dataPreprocessorForAvgScore():
    pass

def dataPreprocessorForGenderedWords(compare_list, title):
    pass