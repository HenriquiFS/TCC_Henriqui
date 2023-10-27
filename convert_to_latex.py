import pandas as pd

def formatar_coluna(dataframe, nome_coluna):
    if nome_coluna not in dataframe.columns:
        raise ValueError(f"A coluna '{nome_coluna}' não existe no DataFrame.")
    
    dataframe[nome_coluna] = dataframe[nome_coluna].apply(lambda x: '{:.3f}'.format(x))
    return dataframe

df = pd.read_excel('teste_com_comentarios_gerados.xlsx')

# df = formatar_coluna(df, 'F1 Score')
# df = formatar_coluna(df, 'Acurácia B.')
# df = formatar_coluna(df, 'R. A. S.')
# df = formatar_coluna(df, 'Precisão')
# df = formatar_coluna(df, 'Recall')

df.to_latex('teste_com_comentarios_gerados7.tex', index=False)