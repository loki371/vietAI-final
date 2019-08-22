

def split_frontal_and_lateral(original_file):
    frontal_lines = []
    lateral_lines = []
    with open(original_file) as f:
        l = '12'
        while(len(l) != 0):
            l = f.readline()
            if l.find('Frontal') != -1:
                frontal_lines.append(l)
            else:
                lateral_lines.append(l)
    
    ## Get file name
    fname = original_file.split('.')[0]
    with open(fname + '_frontal.csv', 'w') as fout:
        fout.writelines(frontal_lines)
    
    with open(fname + '_lateral.csv', 'w') as fout:
        fout.writelines(lateral_lines)
    print(fname)
    print(f'Frontal: {len(frontal_lines)}')
    print(f'Lateral: {len(lateral_lines)}')

if __name__ == '__main__':
    list_file = ['csv/trainPre.csv', 'csv/valPre.csv', 'csv/test.csv']
    for fname in list_file:
        split_frontal_and_lateral(fname)