
from classif import ExtraDataClassifierSimple


def printPerformance(pfm):
    print("-----------")
    for key,val in pfm.items():
        if isinstance(val, dict):
            for _key, _val in val.items():
                print("\t", _key,":",_val)
        else:
            print(key,":",val)

    print("-----------")

def printBestFeatures(topft):
    # topft is a list of dictionaries with feature, p keys
    for feature in topft:
        for k,v in feature.items():
            print(k,v, end='; ')
        
        print()


if __name__=="__main__":

    ed = ExtraDataClassifierSimple(vect='ngram')

    cost_center_pfm = ed.benchmark('cost_center')
    nature_pfm = ed.benchmark('nature')
    # Some metrics are ill-defined (set to default) because some labels in test set dont appear in the predictions made,
    # hence they cannot be computed.
    
    print("\nPERFORMANCE")
    print("\nNature:")
    printPerformance(nature_pfm)
    print("Cost_center:")
    printPerformance(cost_center_pfm)


    print("BEST FEATURES")
    nature_topft = ed.show_top_features('nature')
    cost_center_topft = ed.show_top_features('cost_center')
    print("================== NATURE")
    printBestFeatures(nature_topft)
    print("================== COST CENTER")
    printBestFeatures(cost_center_topft)