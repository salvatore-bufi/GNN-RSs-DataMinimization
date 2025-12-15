import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
## File per recupero di parti mancanti di data characteristic ( unire i risultati )
def diff_matrix_equal_different(df1: pd.DataFrame,
                                df2: pd.DataFrame,
                                keys: list) -> pd.DataFrame:
    """
    Ritorna un df (df4) con:
      - SOLO le righe (chiavi) che hanno almeno una differenza fra df1 e df2
      - SOLO le colonne chiave + le colonne non-chiave che differiscono in almeno una riga
      - Celle non-chiave con 'different' o 'equal' per quella riga-chiave

    Note:
    - Confronta SOLO le colonne comuni non-chiave (presenti in entrambi i DF).
    - Se una riga è presente solo in df1 o solo in df2, quella riga è marcata 'different'
      per tutte le colonne selezionate.
    - In presenza di duplicati sulle chiavi, tiene l'ultima occorrenza in ciascun DF.
    """

    # 1) Normalizza: tieni una sola riga per chiave in ciascun DF
    left  = df1.drop_duplicates(subset=keys, keep='last').copy()
    right = df2.drop_duplicates(subset=keys, keep='last').copy()

    # 2) Colonne comuni non-chiave da confrontare
    common_nonkey = [c for c in left.columns if c in right.columns and c not in keys]
    if not common_nonkey:
        # Nessuna colonna comparabile: ritorna df vuoto con solo chiavi
        return pd.DataFrame(columns=keys).astype({k: left[k].dtype for k in keys})

    # 3) Merge outer per includere anche chiavi presenti in uno solo
    merged = left[keys + common_nonkey].merge(
        right[keys + common_nonkey],
        on=keys,
        how='outer',
        suffixes=('_df1', '_df2'),
        indicator=True
    )

    # 4) Costruisci mask di differenza per ciascuna colonna
    col_diff_masks = {}
    for c in common_nonkey:
        a = merged[f'{c}_df1']
        b = merged[f'{c}_df2']
        # equal se uguali o entrambi NaN
        equal = a.eq(b) | (a.isna() & b.isna())
        # Se la riga è presente solo da una parte, consideriamo 'different'
        diff = ~equal
        col_diff_masks[c] = diff

    # 5) Righe con QUALSIASI differenza (oppure riga presente solo in uno dei DF)
    row_any_diff = np.zeros(len(merged), dtype=bool)
    for c in common_nonkey:
        row_any_diff |= col_diff_masks[c].values
    # Se riga non è in entrambi, tutto è "different"
    row_only_left  = (merged['_merge'] == 'left_only').values
    row_only_right = (merged['_merge'] == 'right_only').values
    row_any_diff |= row_only_left | row_only_right

    # Se nessuna riga differisce, ritorna df vuoto con chiavi
    if not row_any_diff.any():
        return pd.DataFrame(columns=keys).astype({k: left[k].dtype for k in keys})

    # 6) Seleziona SOLO le colonne con almeno una differenza in qualche riga
    cols_with_diff = []
    for c in common_nonkey:
        col_has_diff = (col_diff_masks[c] | (merged['_merge'] != 'both')).any()
        if col_has_diff:
            cols_with_diff.append(c)

    # 7) Costruisci l'output con chiavi + SOLO le colonne con differenze
    out = merged.loc[row_any_diff, keys].copy()
    in_both = (merged.loc[row_any_diff, '_merge'] == 'both').values

    for c in cols_with_diff:
        diff_mask = col_diff_masks[c].loc[row_any_diff].values
        # dove la riga è in entrambi e non differisce -> 'equal'; altrimenti 'different'
        status = np.where(in_both & (~diff_mask), 'equal', 'different')
        out[c] = status

    # Ordina per chiavi (opzionale) e resetta indice
    out = out.sort_values(keys).reset_index(drop=True)
    return out


# --- Esempio d'uso ---
# df4 = diff_matrix_equal_different(df1, df2, keys=['dataset_name', 'strategy', 'cutoff'])
# df4.to_csv('diff_report.tsv', sep='\t', index=False)

def diff_rows(df1: pd.DataFrame, df2: pd.DataFrame, keys=None) -> pd.DataFrame:
    """
    Ritorna un DataFrame con le righe diverse tra df1 e df2.
    - Se keys è fornito (lista di colonne chiave):
        * include righe presenti solo in df1 o solo in df2
        * e righe con stesse chiavi ma valori diversi su almeno una colonna non-chiave.
      L'output contiene le chiavi e, per ogni colonna confrontata, le coppie <col>_df1 / <col>_df2.
      Aggiunge la colonna _where ∈ {'left_only','right_only','different'}.
    - Se keys è None:
        * usa la differenza simmetrica per riga intera sui campi in comune (stesse colonne),
          restituendo le righe che compaiono solo in uno dei due DF, con colonna _where.
    """
    # Se non specificate, prova ad usare chiavi standard se presenti in entrambi
    if keys is None:
        candidate = ['dataset_name', 'strategy', 'cutoff']
        if all(k in df1.columns for k in candidate) and all(k in df2.columns for k in candidate):
            keys = candidate

    if keys:  # --- MODALITÀ CON CHIAVI ---
        # Allinea le colonne confrontabili (chiavi + colonne comuni non-chiave)
        common_cols = [c for c in df1.columns if c in df2.columns]
        nonkey_common = [c for c in common_cols if c not in keys]
        # Riduci ai soli campi rilevanti e rimuovi duplicati per chiave (tiene l'ultimo)
        left  = df1[keys + nonkey_common].drop_duplicates(subset=keys, keep='last').copy()
        right = df2[keys + nonkey_common].drop_duplicates(subset=keys, keep='last').copy()

        merged = left.merge(
            right, on=keys, how='outer', suffixes=('_df1', '_df2'), indicator=True
        )

        # Righe presenti solo in uno dei due
        mask_left_only  = merged['_merge'] == 'left_only'
        mask_right_only = merged['_merge'] == 'right_only'

        # Righe presenti in entrambi ma con differenze in almeno una colonna non-chiave
        mask_both = merged['_merge'] == 'both'
        if nonkey_common:
            diff_mask = pd.Series(False, index=merged.index)
            for c in nonkey_common:
                a = merged[f'{c}_df1']
                b = merged[f'{c}_df2']
                equal = a.eq(b) | (a.isna() & b.isna())
                diff_mask |= ~equal
            mask_diff = mask_both & diff_mask
        else:
            # Se non ci sono colonne non-chiave comuni, nessuna differenza di valori da segnalare
            mask_diff = pd.Series(False, index=merged.index)

        out = merged.loc[mask_left_only | mask_right_only | mask_diff].copy()
        out.loc[mask_left_only,  '_where'] = 'left_only'
        out.loc[mask_right_only, '_where'] = 'right_only'
        out.loc[mask_diff,       '_where'] = 'different'

        # Ordina colonne: chiavi, _where, poi coppie <col>_df1/<col>_df2
        pair_cols = sum(([f'{c}_df1', f'{c}_df2'] for c in nonkey_common), [])
        ordered_cols = keys + ['_where'] + pair_cols
        # Mantieni anche eventuali colonne non comuni (se vuoi: togli questa parte per output più compatto)
        rest = [c for c in out.columns if c not in ordered_cols + ['_merge']]
        return out[ordered_cols + rest].reset_index(drop=True)

    else:  # --- MODALITÀ SENZA CHIAVI: differenza simmetrica per riga intera ---
        common_cols = [c for c in df1.columns if c in df2.columns]
        d1 = df1[common_cols].copy()
        d2 = df2[common_cols].copy()

        d1['_where'] = 'left_only'
        d2['_where'] = 'right_only'

        # Concatena e rimuovi le righe duplicate presenti in entrambi (keep=False elimina tutte le occorrenze duplicate)
        both = pd.concat([d1, d2], ignore_index=True)
        # Per confrontare riga intera, usiamo tutte le colonne comuni tranne il flag
        subset_cols = [c for c in common_cols]  # tutte le colonne comuni
        mask_unique = ~both.duplicated(subset=subset_cols, keep=False)
        out = both.loc[mask_unique].reset_index(drop=True)
        return out

def overwrite_most_recent(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # Chiavi di join e colonne da sovrascrivere
    keys = ['dataset_name', 'strategy', 'cutoff']
    recent_cols = ['most_recent_1', 'most_recent_2', 'most_recent_3']

    # 1) Teniamo solo le colonne necessarie e rimuoviamo eventuali duplicati in df2 sulle chiavi
    df2_clean = df2.copy()

    # 2) (opzionale) forza numerico sulle colonne recent per evitare problemi di dtype
    for c in recent_cols:
        if c in df2_clean.columns:
            df2_clean[c] = pd.to_numeric(df2_clean[c], errors='coerce')

    # 3) Merge left: preserva ordine e righe di df1
    df3 = df1.merge(df2_clean, on=keys, how='left', suffixes=('', '_new'))

    # 4) Sovrascrive valori di df1 con quelli di df2 quando presenti (non NaN)
    for c in recent_cols:
        c_new = f'{c}_new'
        if c_new in df3.columns:
            df3[c] = np.where(df3[c_new].notna(), df3[c_new], df3[c])
            df3.drop(columns=[c_new], inplace=True)

    # df3 ha stessa lunghezza di df1 e contiene i valori corretti dove disponibili
    return df3

# Esempio d'uso:
df1 = pd.read_csv('/home/salvatore/GraphDataMinimization/Dataset_Characteristics.tsv', sep='\t')
# df2 = pd.read_csv('/home/salvatore/GraphDataMinimization/Dataset_Characteristics_amazontimestamp.tsv', sep='\t')
df2 = pd.read_csv('/home/salvatore/GraphDataMinimization/Data_Char_amztimestamp_full.tsv', sep='\t')
df3 = overwrite_most_recent(df1, df2)
# df4 = diff_rows(df1, df3)
df4 = diff_matrix_equal_different(df1, df3, keys=['dataset_name', 'strategy', 'cutoff'])
print()
df3.to_csv('Dataset_Characteristics_fixed_1.tsv', sep='\t', index=False)