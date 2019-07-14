import tokenization
import pandas as pd

f = open('ProData\\2019-05-14.txt', 'w', encoding='utf-8')
input_file = "ProData\\2019-05-14-18-53.xls"
df = pd.read_excel(input_file)
str_list = list(df['Requirement'])
basic_tokenizer = tokenization.BasicTokenizer(do_lower_case=False)
id = 10000
for req in str_list:

    tok_list = basic_tokenizer.tokenize(req)
    f.write('id:%d, \n' % (id))
    for tok in tok_list:
        f.write(tok+",\n")
    f.write('\n')
    id += 1

f.close()


f = open('ProData\\0610-predict_ipadusa.txt', 'w', encoding='utf-8')

input_file = "ProData\\0608-ipadusa_needs_1.xlsx"
df = pd.read_excel(input_file)
str_list = list(df['Content'])
basic_tokenizer = tokenization.BasicTokenizer(do_lower_case=False)
id = 10000

for req in str_list:
    tok_list = basic_tokenizer.tokenize(req)
    f.write('id:%d, \n' % (id))
    for tok in tok_list:
        f.write(tok+",\n")
    f.write('\n')
    id += 1

f.close()
