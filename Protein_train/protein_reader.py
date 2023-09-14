import openpyxl 


def get_protein_data(path='protein_sequences_full.xlsx'):
    workbook = openpyxl.load_workbook(path)
    sheet = workbook.active
    protein_names = []
    for cell in sheet['A']:
        protein_names.append(cell.value)
    gene_names = []
    for cell in sheet['B']:
        gene_names.append(cell.value)
    sequences = []
    for i, cell in enumerate(sheet['C']):
        sequences.append(cell.value)
    return protein_names, gene_names, sequences
