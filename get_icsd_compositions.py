# %%
import os
import warnings

import pandas as pd
from pymatgen.core import Structure
from pymatviz import count_elements, ptable_heatmap_plotly
from tqdm import tqdm

__author__ = "Janosh Riebesell"
__date__ = "2023-05-02"

ICSD_PATH = "/Users/janosh/dev/icsd/icsd_cifs"

# %%
with open("icsd_full_data_unique_no_frac_no_penta_2020_icsd_codes_only.txt", "r") as f:
    synthnn_train_ids = f.read().splitlines()


# %%
avail_icsd_ids = [
    filename.replace(".cif", "").replace("icsd_", "")
    for filename in os.listdir(ICSD_PATH)
]

# %%
assert len({*synthnn_train_ids}) == len(
    synthnn_train_ids
), "icsd_codes contains duplicates"
assert len({*avail_icsd_ids}) == len(
    avail_icsd_ids
), "avail_icsd_ids contains duplicates"


# %%
intersect = {*synthnn_train_ids} & {*avail_icsd_ids}

print(f"{len(avail_icsd_ids)=:,}")
print(f"{len(synthnn_train_ids)=:,}")
print(f"{len(intersect)=:,}")
# on 2023-05-02
# len(avail_icsd_ids)=189,371
# len(synthnn_train_ids)=53,594
# len(intersect)=30,168


# %% load structures for IDs in intersect
def read_cif(cif_path: str) -> Structure | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            struct = Structure.from_file(cif_path)
    except (
        ValueError,  # Invalid cif file with no structures!
        AssertionError,  # assert len(items) % n == 0
        ZeroDivisionError,  # modulo by zero in assert len(items) % n == 0
    ):
        return None
    return struct


structures: dict[str, Structure] = {}
for icsd_id in tqdm(intersect):
    struct = read_cif(f"{ICSD_PATH}/icsd_{icsd_id}.cif")
    if struct:
        structures[icsd_id] = struct


# %% get composition for each structure
compositions = {
    icsd_id: struct.composition.alphabetical_formula
    for icsd_id, struct in structures.items()
}

elem_counts = count_elements(list(compositions.values()), count_mode="occurrence")


# %% plot elemental occurrence heatmap
fig = ptable_heatmap_plotly(elem_counts[elem_counts > 1], log=True)
fig.layout.title = "Elemental occurrence of available SynthNN ICSD training data"
fig.show()


# %% write IDs with their composition back to disk as CSV
id_col = "icsd_id"
df_avail_formulas = pd.DataFrame(compositions.items(), columns=[id_col, "composition"])

df_avail_formulas.set_index(id_col).to_csv("avail-icsd-compositions.csv")
