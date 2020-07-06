# %%

from deeppavlov import configs
from deeppavlov.core.common.file import read_json

config = read_json(configs.go_bot.gobot_simple_dstc2)

#%%
from deeppavlov import build_model

model = build_model(config)

#%%