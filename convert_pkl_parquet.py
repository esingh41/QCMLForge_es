import ast

import numpy as np
import pandas as pd
import qcelemental as qcel


def _parse_1d_array(value, dtype):
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    text = str(value).replace('[', ' ').replace(']', ' ').replace(',', ' ')
    return np.fromstring(text, sep=' ', dtype=dtype)


def _parse_2d_coords(value):
    arr = _parse_1d_array(value, dtype=float)
    return arr.reshape(-1, 3)


def _parse_indices(value):
    if isinstance(value, np.ndarray):
        return value.astype(int, copy=False).tolist()
    if isinstance(value, (list, tuple)):
        return [int(i) for i in value]
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return [int(i) for i in parsed]
    except (SyntaxError, ValueError):
        pass
    arr = _parse_1d_array(text, dtype=int)
    return arr.tolist()

def pkl_to_parquet(pickle_file):
    df = pd.read_pickle(pickle_file)
    if 'qcel_molecule' in df.columns:
        df["qcel_molecule"] = df["qcel_molecule"].apply(lambda x: x.to_string('psi4'))
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str)
    root = pickle_file.rpartition(".")[0] 
    df.to_parquet(f'{root}.parquet')

def parquet_to_pkl(parquet_file):
    df = pd.read_parquet(parquet_file)
    if 'qcel_molecule' in df.columns:
        df['qcel_molecule'] = df['qcel_molecule'].apply(lambda x : qcel.models.Molecule.from_string(x, fix_com=True, fix_orientation=True))
    if 'ZA' in df.columns:
        df['ZA'] = df['ZA'].apply(lambda x: _parse_1d_array(x, dtype=int))
    if 'RA' in df.columns:
        df['RA'] = df['RA'].apply(_parse_2d_coords)
    if 'ZB' in df.columns:
        df['ZB'] = df['ZB'].apply(lambda x: _parse_1d_array(x, dtype=int))
    if 'RB' in df.columns:
        df['RB'] = df['RB'].apply(_parse_2d_coords)
    if 'Frag1_indices' in df.columns:
        df['Frag1_indices'] = df['Frag1_indices'].apply(_parse_indices)
    if 'Frag2_indices' in df.columns:
        df['Frag2_indices'] = df['Frag2_indices'].apply(_parse_indices)

    root = parquet_file.rpartition(".")[0]
    df.to_pickle(f"{root}.pkl")
    return df


#How do I keep track of the old data types though is my question
def main():
    df = pd.read_pickle("small_189K_saptpbe0-d4_totals_train.pkl").head(20)
    print(df.info())
    print(df.dtypes)
    row = df.iloc[0]
    print(f"{row['ZA'] = }")
    print(f"{type(row['ZA']) = }")
    print(f"{row['RA'] = }")
    print(f"{type(row['RA']) = }")
    print(f"{row['Frag1_indices'] = }")
    print(f"{type(row['Frag1_indices']) = }")
    print(f"{row['Frag1_indices'] = }")
    print(f"{type(row['Frag1_indices']) = }")
    # print(df[["ZA", "ZB"]].to_string())
    return
    # df.to_pickle("small_189K_saptpbe0-d4_totals_train.pkl")
    # return
    pkl_to_parquet("small_189K_saptpbe0-d4_totals_train.pkl")
    return


if __name__ == "__main__":
    main()
