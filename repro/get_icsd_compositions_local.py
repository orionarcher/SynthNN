# %%
import os
import warnings

import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm

__author__ = "Janosh Riebesell"
__date__ = "2023-05-02"

ICSD_PATH = "/Users/janosh/dev/icsd/icsd_cifs"

# %%
train_ids_path = "icsd_full_data_unique_no_frac_no_penta_2020_icsd_codes_only.txt"
with open(train_ids_path) as file:
    synthnn_train_ids = list(map(int, file.read().splitlines()))


# %%
avail_icsd_ids = [
    int(filename.replace(".cif", "").replace("icsd_", ""))
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
missing = [*{*synthnn_train_ids} - {*avail_icsd_ids}]

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


# %% write IDs with their composition back to disk as CSV
compositions = {
    icsd_id: struct.composition.alphabetical_formula
    for icsd_id, struct in structures.items()
}
id_col = "icsd_id"
df_avail_formulas = pd.DataFrame(
    compositions.items(), columns=[id_col, "composition"]
).set_index(id_col)
df_avail_formulas["n_atoms"] = [len(struct) for struct in structures.values()]
df_avail_formulas["n_elements"] = [len(comp) for comp in compositions.values()]

df_avail_formulas.to_csv("avail-icsd-from-local.csv")
