from utils.dataloader import DataLoader
import pandas as pd
from settings.constants import TRAIN_CSV

df = pd.read_csv(TRAIN_CSV, header=0)

dl = DataLoader()
dl.fit(df)
print(dl.load_data())