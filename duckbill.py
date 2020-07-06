# %%

from deeppavlov import configs
from deeppavlov.core.common.file import read_json

config = read_json(configs.go_bot.gobot_simple_dstc2)

#%%
from deeppavlov import build_model, evaluate_model, train_model

model = train_model(config, download=True)

#%%