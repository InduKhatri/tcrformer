import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def create_data(path='data/vdjdb-2021-09-05/vdjdb.txt', epi ='CMV pp65 NLVPMVATV'):
    df = pd.read_csv(path, sep='\t')
    df_vdj_sel = pd.DataFrame()
    df = df.dropna(axis=0, how='any')
    df_vdj_sel['complex.id'] = df['complex.id']
    df_vdj_sel['Gene'] = df['gene']
    df_vdj_sel['CDR3'] = df['cdr3']
    df_vdj_sel['V'] = df['v.segm']
    df_vdj_sel['J'] = df['j.segm']
    df_vdj_sel['Species'] = df['species']
    df_vdj_sel['MHC A'] = df['mhc.a']
    df_vdj_sel['MHC B'] = df['mhc.b']
    df_vdj_sel['MHC class'] = df['mhc.class']
    df_vdj_sel['Epitope'] = df['antigen.epitope']
    df_vdj_sel['Epitope gene'] = df['antigen.gene']
    df_vdj_sel['Epitope species'] = df['antigen.species']
    df_vdj_sel['Score'] = df['vdjdb.score']

    # df_vdj_sel = df_vdj_sel.dropna(axis=0, how='any')
    df_vdj_sel = df_vdj_sel.loc[~df_vdj_sel['Species'].isin(["MacacaMulatta"])]
    df_vdj_sel = df_vdj_sel.loc[~df_vdj_sel['Score'].isin([0])]

    df_vdj_sel = df_vdj_sel.loc[~df_vdj_sel['Epitope species'].isin(
        ["PseudomonasAeruginosa", "PseudomonasFluorescens", "SaccharomycesCerevisiae", "HomoSapiens", "synthetic",
         "GallusGallus", "TriticumAestivum", "ManducaSexta", "Synthetic", "HHV"])]
    df_vdj_sel = df_vdj_sel.drop_duplicates(subset=['Gene','CDR3', 'V', 'J', 'MHC A', 'MHC B'], keep='first')

    df_vdj_sel['labels'] = df_vdj_sel['Epitope species'].astype(str) + " " + df_vdj_sel['Epitope gene'].astype(str) + " " + \
                           df_vdj_sel['Epitope'].astype(str)
    # Fix Genes
    df_vdj_sel['V'] = df_vdj_sel['V'].str.split('*').str.get(0)
    df_vdj_sel['J'] = df_vdj_sel['J'].str.split('*').str.get(0)

    # Magic
    # df_vdj_sel['length'] = df_vdj_sel.CDR3.str.len()
    # df_vdj_sel = df_vdj_sel[df_vdj_sel.length > 10]
    # del df_vdj_sel['length']
    df_vdj_sel = df_vdj_sel.loc[df_vdj_sel['Gene'].isin(["TRB"])]
    df_vdj_sel = df_vdj_sel[df_vdj_sel.groupby('labels')['labels'].transform('count').ge(50)]
    labels = df_vdj_sel['labels'].value_counts().to_dict()
    if epi is not None:
        labels = df_vdj_sel['labels'].value_counts().to_dict()
        df['labels'].where(df['labels'] == epi, -1, inplace=True)
        df["labels"].replace({epi: 1}, inplace=True)
        if df['labels'].value_counts().array.__len__() == 2 and df['labels'].value_counts().values[-1] == labels[epi]:
            df.to_csv('data/epitope_{}.csv'.format(epi.replace(" ", "-")), encoding='utf-8', index=False)
            df = df_vdj_sel[df_vdj_sel.groupby('labels')['labels'].transform('count').ge(50)]
            print(f"File saved in ProtTrans/data/epitope_{epi.replace(' ', '-')}.csv")
        else:
            df = df_vdj_sel[df_vdj_sel.groupby('labels')['labels'].transform('count').ge(50)]
            print("Error")

def df_ab(df_ab):
    df_paired = df_ab[df_ab.groupby('complex.id')['complex.id'].transform('count').eq(2)]
    df_unpaired = df_ab[~df_ab.groupby('complex.id')['complex.id'].transform('count').eq(2)]

    df1 = df_paired.loc[df_paired['Gene'].isin(["TRB"])]
    df2 = df_paired.loc[df_paired['Gene'].isin(["TRA"])]
    combo = pd.merge(df1, df2, on="complex.id")
    combo.drop(['Epitope species_y', 'Epitope gene_y', 'Epitope_y', 'MHC class_y', 'MHC B_y', 'MHC A_y', 'Species_y', 'Score_y', 'labels_y'], axis=1, inplace=True)
    combo.rename(columns={'Gene_x': 'Gene', 'CDR3_x': 'CDR3'}, inplace=True)
    combo.rename(columns={'V_x': 'V', 'J_x': 'J'}, inplace=True)
    combo.rename(columns={'Species_x': 'Species', 'MHC A_x': 'MHC A'}, inplace=True)
    combo.rename(columns={'MHC B_x': 'MHC B', 'MHC class_x': 'MHC class', 'Epitope_x': 'Epitope',
                          'Epitope gene_x': 'Epitope gene', 'Epitope species_x': 'Epitope species', 'Score_x': 'Score',
                          'labels_x': 'labels'}, inplace=True)
    VTRA = ['TRAV1', 'TRAV1-1', 'TRAV1-2', 'TRAV10', 'TRAV10N', 'TRAV11', 'TRAV12-1', 'TRAV12-2', 'TRAV12-3', 'TRAV12D-1', 'TRAV12N-2', 'TRAV12N-3', 'TRAV13-1', 'TRAV13-2', 'TRAV13N-2', 'TRAV14-1', 'TRAV14-2', 'TRAV14/DV4', 'TRAV14D-1', 'TRAV14D-3/DV8', 'TRAV14N-1', 'TRAV14N-2', 'TRAV14N-3', 'TRAV15-1/DV6-1', 'TRAV15N-1', 'TRAV16', 'TRAV16D/DV11', 'TRAV16N', 'TRAV17', 'TRAV18', 'TRAV19', 'TRAV2', 'TRAV20', 'TRAV21', 'TRAV21/DV12', 'TRAV22', 'TRAV24', 'TRAV25', 'TRAV26-1', 'TRAV26-2', 'TRAV27', 'TRAV29/DV5', 'TRAV3', 'TRAV3-1', 'TRAV3-3', 'TRAV3-4', 'TRAV30', 'TRAV34', 'TRAV35', 'TRAV36/DV7', 'TRAV38-1', 'TRAV38-2/DV8', 'TRAV3D-3', 'TRAV3N-3', 'TRAV4', 'TRAV4-2', 'TRAV41', 'TRAV4D-3', 'TRAV4D-4', 'TRAV4N-3', 'TRAV4N-4', 'TRAV5', 'TRAV5-1', 'TRAV5N-4', 'TRAV6', 'TRAV6-1', 'TRAV6-2', 'TRAV6-3', 'TRAV6-4', 'TRAV6-5', 'TRAV6-6', 'TRAV6-7/DV9', 'TRAV6D-3', 'TRAV6D-4', 'TRAV6D-5', 'TRAV6D-6', 'TRAV6N-5', 'TRAV6N-6', 'TRAV6N-7', 'TRAV7-1', 'TRAV7-2', 'TRAV7-3', 'TRAV7-4', 'TRAV7-5', 'TRAV7D-2', 'TRAV7D-3', 'TRAV7D-4', 'TRAV7D-5', 'TRAV7N-4', 'TRAV7N-5', 'TRAV7N-6', 'TRAV8-2', 'TRAV8-3', 'TRAV8-4', 'TRAV8-6', 'TRAV8D-1', 'TRAV8D-2', 'TRAV8N-2', 'TRAV9-1', 'TRAV9-2', 'TRAV9N-2', 'TRAV9N-3', 'TRAV9N-4']
    JTRA = ['TRAJ10', 'TRAJ11', 'TRAJ12', 'TRAJ13', 'TRAJ15', 'TRAJ16', 'TRAJ17', 'TRAJ18', 'TRAJ2', 'TRAJ20', 'TRAJ21', 'TRAJ22', 'TRAJ23', 'TRAJ24', 'TRAJ26', 'TRAJ27', 'TRAJ28', 'TRAJ29', 'TRAJ3', 'TRAJ30', 'TRAJ31', 'TRAJ32', 'TRAJ33', 'TRAJ34', 'TRAJ36', 'TRAJ37', 'TRAJ38', 'TRAJ39', 'TRAJ4', 'TRAJ40', 'TRAJ41', 'TRAJ42', 'TRAJ43', 'TRAJ44', 'TRAJ45', 'TRAJ47', 'TRAJ48', 'TRAJ49', 'TRAJ5', 'TRAJ50', 'TRAJ52', 'TRAJ53', 'TRAJ54', 'TRAJ56', 'TRAJ57', 'TRAJ58', 'TRAJ6', 'TRAJ8', 'TRAJ9']
    combo['Gene'] = combo['Gene'].astype(str) + " [SEP] " + combo['Gene_y']

    df_exp_2 = pd.concat([df_unpaired, combo], ignore_index=True)
    df_exp_2 = df_exp_2.sample(frac=1, random_state=44).reset_index(drop=True)
    tokens = []
    for i in VTRA:
        a = ''.join("[{}]".format(i))
        tokens.append(a)

def create_frame(path='data/vdjdb-2021-09-05/vdjdb.txt', ul_label_count=None, ll_label_count=100, min_sequence_length=10,
                 max_sequence_length=None):
    df = pd.read_csv(path, sep='\t')
    df = df.dropna(axis=0, how='any')
    df = df.loc[~df['Species'].isin(["MacacaMulatta"])]
    df = df.loc[df['Gene'].isin(["TRB"])]
    df = df.loc[~df['Score'].isin([0])]
    df = df.loc[~df['Epitope species'].isin(
        ["PseudomonasAeruginosa", "PseudomonasFluorescens", "SaccharomycesCerevisiae", "HomoSapiens", "synthetic",
         "GallusGallus", "TriticumAestivum", "ManducaSexta", "Synthetic", "HHV"])]
    # Create Labels
    df['labels'] = df['Epitope'].astype(str) + " " + df['Epitope gene'].astype(str) + " " + df[
        'Epitope species'].astype(str)
    df = df.drop_duplicates(subset=['CDR3', 'V', 'J', 'MHC A', 'MHC B', 'labels'], keep='first')

    # measurer = np.vectorize(len)
    # res1 = dict(zip(df, measurer(df.values.astype(str)).max(axis=0)))
    ## Max CDR3 length = 38, Min Length = 4
    df['length'] = df.CDR3.str.len()
    if max_sequence_length is not None:
        df = df[df.length < max_sequence_length]
    df = df[df.length > 10]
    del df['length']

    # Remove Columns unrelated to CDR3
    #
    # df['labels'].value_counts().plot(kind='bar')
    # plt.title('Epitope species', fontdict={'fontsize': 20})
    # plt.tight_layout()
    # plt.show()

    # # Remove all Species other than Humans and Mouse
    # df = df.loc[df['Species'].isin(["HomoSapiens", "MusMusculus"])]

    # Remove CDR3 samples with length less than 5

    # Remove based on Label occurrence
    if ul_label_count is not None:
        df = df[df.groupby('labels')['labels'].transform('count').le(ul_label_count)]
    df = df[df.groupby('labels')['labels'].transform('count').ge(ll_label_count)]

    del df['Epitope']
    del df['Epitope gene']
    del df['Epitope species']
    return df


def combine(df, sec_df, final_df, query):
    """
    :param df: vdj
    :param sec_df: other_source_df
    """
    # df_vdj_sel = pd.DataFrame()
    # df_dash_sel = pd.DataFrame()
    df_beta = final_df
    query = query
    df = df
    df_dash_human_pp65 = sec_df
    df_vdj_sel = pd.DataFrame()
    df_dash_sel = pd.DataFrame()

    df_pp65 = df.loc[df['Epitope gene'].isin([query])]

    df_vdj_sel['CDR3'] = df_pp65['CDR3']
    df_vdj_sel['V'] = df_pp65['V']
    df_vdj_sel['J'] = df_pp65['J']
    df_vdj_sel['labels'] = df_pp65['Epitope'].astype(str) + " " + df_pp65['Epitope gene'].astype(str) + " " + df_pp65[
        'Epitope species'].astype(str)

    df_dash_sel['CDR3'] = df_dash_human_pp65['beta']
    df_dash_sel['V'] = df_dash_human_pp65['v_beta']
    df_dash_sel['J'] = df_dash_human_pp65['j_beta']
    df_dash_sel['labels'] = 'pp65 CMV'

    df_out = pd.concat([df_vdj_sel, df_dash_sel])
    df_beta = pd.concat([df_beta, df_out])

    return df_beta

def split_data(df, fnc=None, save=False, size=0.20):
    train, test = train_test_split(df, stratify=df['labels'], test_size=size)
    if save:
        test.to_csv('data/test_{}.csv'.format(fnc), encoding='utf-8', index=False)
        train.to_csv('data/train_{}.csv'.format(fnc), encoding='utf-8', index=False)
    return train, test
