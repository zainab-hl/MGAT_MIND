class Config:
    DATA_PATH = "/kaggle/input/datasets/zainabhalhoul/mindlarge"
    EDGE_PATH = "/kaggle/input/datasets/zainabhalhoul/mindlarge-edges"
    
    TEXT_DIM = 128
    ENTITY_DIM = 100
    CATEGORY_DIM = 32
    HIDDEN_DIM = 128
    ID_DIM = 32
    
    BATCH_SIZE = 1024        # increased from 256 — fills both GPUs 
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 3
    NUM_NEGATIVES = 4
    NUM_WORKERS = 4          # parallel data loading
    PREFETCH_FACTOR = 2      # prefetch batches ahead of time

config = Config()