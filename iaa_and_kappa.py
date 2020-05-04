# need to install python-bs4 and lxml

from bs4 import BeautifulSoup
import re
import os
import numpy as np
from collections import defaultdict


def judgment_list(dir):
    tag_division = {}
    for file in os.listdir(dir):
        if file.endswith(".xml"):
            # parse the annotator's file
            file = open(os.path.join(dir, file), encoding='utf-8')
            contents = file.read()
            file.close()
            soup = BeautifulSoup(contents, 'xml')

            tags = soup.find('TAGS')
            jokes = soup.find_all('TEXT')

            # get the tag/item number pairs
            tags = str(tags).split("\n")
            regex = re.compile(r"^<(\w*).*(ITEM +\d*)")

            for count in range(1, len(tags)-1):
                tag_and_id = regex.findall(tags[count])
                tag_division[tag_and_id[0][1].strip()] = tag_and_id[0][0]

    return tag_division


def agreement(one, two):
    categories = {"wordplay": 0, "character": 1, "shock": 2, "focus": 3, "reference": 4, "total": 5}
    agreements = np.zeros((len(categories), len(categories)))

    # find the jokes they agreed on
    for joke in one.keys():
        if two.get(joke) is not None:
            agreements[categories[one[joke]]][categories[two[joke]]] = agreements[categories[one[joke]]][categories[two[joke]]] + 1
            agreements[categories[one[joke]]][categories["total"]] = agreements[categories[one[joke]]][categories["total"]] + 1
            agreements[categories["total"]][categories[two[joke]]] = agreements[categories["total"]][categories[two[joke]]] + 1
            agreements[categories["total"]][categories["total"]] = agreements[categories["total"]][categories["total"]] + 1

    # calculate expected agreement
    A_e_k = 0
    for x in range(0, len(categories) - 1):
        A_e_k = A_e_k + ((agreements[x][categories["total"]] / agreements[categories["total"]][categories["total"]]) * (agreements[categories["total"]][x] / agreements[categories["total"]][categories["total"]]))

    # calculate observed agreement
    A_o = 0
    for x in range(0, len(categories) - 1):
        A_o = A_o + (agreements[x][x] / agreements[categories["total"]][categories["total"]])

    # calculate and display agreement metrics
    kappa = (A_o - A_e_k) / (1 - A_e_k)

    print("Cohen kappa:", round(kappa, 4))
    print("Inter-Annotator Agreement:", round(A_o, 4))
    print()
    print("NOTE: Due to how we divided up our annotation, precision and recall are the same as IAA.")
    print("This is because each annotator mandatorily highlighted the same text ('ITEM ###'), and each")
    print("joke was annotated by two people, so the number of shared annotations for each person match.")
    print()
    print("Judgment Comparison")
    print("Axes: wordplay, character, shock, focus, reference, total")
    print(agreements)
    print()

    return agreements[5], agreements[:,5], np.diagonal(agreements), np.diagonal(agreements)

if __name__ == '__main__':
    dom = judgment_list("Dom\Finished")
    james = judgment_list("James\Finished")
    miriam = judgment_list("Miriam\Finished")

    print()
    print("ANNOTATORS ONE AND TWO")
    agreement(dom, james)
    print()
    print("ANNOTATORS ONE AND THREE")
    agreement(dom, miriam)
    print()
    print("ANNOTATORS TWO AND THREE")
    agreement(james, miriam)
    print()
