# %%

import pandas as pd
from pymatgen.ext.matproj import MPRester
from pymatviz import count_elements, ptable_heatmap_plotly

from repro.env import legacy_api_key

__author__ = "Janosh Riebesell"
__date__ = "2023-05-02"


# %%

def dict_to_formula(element_dict):
    formula = ""
    for element, count in element_dict.items():
        formula += f"{element}{int(count)}"
    return formula

train_ids_path = "icsd_full_data_unique_no_frac_no_penta_2020_icsd_codes_only.txt"
with open(train_ids_path) as file:
    synthnn_train_ids = list(map(int, file.read().splitlines()))

fields = ["icsd_ids", "formula", "material_id", "nelements", "nsites"]
docs = MPRester(legacy_api_key).query({"icsd_ids": {"$in": synthnn_train_ids}}, fields)
print(f"{len(docs)=:,}")

df_avail = pd.DataFrame(docs).explode("icsd_ids").set_index("icsd_ids")

df_avail["formula"] = df_avail.formula.apply(dict_to_formula)

print(f"{len(df_avail)=:,}")
df_avail = df_avail.loc[list({*synthnn_train_ids} & {*df_avail.index})]
print(f"{len(df_avail)=:,}")
df_avail.to_csv("avail-icsd-from-mp.csv")

df_avail["formula"].to_csv('icsd_positive.txt', header=None, index=None, sep=' ')

# %%

elem_counts = count_elements(df_avail.formula_pretty, count_mode="occurrence")


# %% plot elemental occurrence heatmap
fig = ptable_heatmap_plotly(elem_counts[elem_counts > 1], log=True)
fig.layout.title = "Elemental occurrence of available SynthNN ICSD training data"
fig.show()
