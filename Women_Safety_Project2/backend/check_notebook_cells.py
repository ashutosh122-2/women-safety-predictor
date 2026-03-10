import json,sys
nb='backend/evaluate_model.ipynb'
with open(nb,'r',encoding='utf-8') as f:
    nbj=json.load(f)
code_cells=[c for c in nbj['cells'] if c.get('cell_type')=='code']
print('Total code cells:',len(code_cells))
for i,c in enumerate(code_cells, start=1):
    src='\n'.join(c.get('source',[]))
    try:
        compile(src,'<cell %d>'%i,'exec')
    except Exception as e:
        print('\nSyntax error in code cell',i,':',repr(e))
        print('--- Source (first 400 chars) ---')
        print(src[:400])
        sys.exit(2)
print('\nNo syntax errors detected in code cells (all compiled).')
