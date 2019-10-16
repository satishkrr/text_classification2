import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):

    model_name = 'CharacterModel'
    dataset_name = 'load_movie_sentiment_dataset'
    max_samples  = 1000000

    embed_dim = 100
    hidden_dim = 100
    num_layers = 1
    
    dropout = 0.2

    LR = 0.01
    MOMENTUM=0.1



    
