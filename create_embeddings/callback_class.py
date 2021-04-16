import datetime

from gensim.models.callbacks import CallbackAny2Vec

#creates callback logs for Gensim training
class callback_log(CallbackAny2Vec):
    '''Output loss at each epoch'''
    def __init__(self):
        self.epoch = 1
        self.losses = []
        self.start_time = datetime.datetime.now()

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        print(datetime.datetime.now()-self.start_time)
        self.epoch += 1