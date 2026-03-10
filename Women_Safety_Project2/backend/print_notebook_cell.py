import json,sys
nb='backend/evaluate_model.ipynb'
if len(sys.argv)<2:
    print('Usage: print_notebook_cell.py <cell_index>')
    sys.exit(2)
ci=int(sys.argv[1])
with open(nb,'r',encoding='utf-8') as f:
    nbj=json.load(f)
cells=nbj['cells']
if ci<1 or ci>len(cells):
    print('Index out of range', ci, 'valid range 1..', len(cells))
    sys.exit(2)
cell=cells[ci-1]
print('Cell',ci,'type',cell.get('cell_type'))
print('\n'.join(cell.get('source',[])))
