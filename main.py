#from twitter import TwitterClient
from model import SentimentAnalysis

if __name__ == "__main__":
    #api = TwitterClient()
    model = SentimentAnalysis()
    model.NB_train()
    model.NB_test()
    model.LR_train()
    model.LR_test()
    #model.SVM_train()
    #model.SVM_test()
    #model.RF_train()
    #model.RF_test()
    model.complete_list()
