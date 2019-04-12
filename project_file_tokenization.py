import tokenization
import pandas as pd

f = open('ProData\\2019-04-12.txt', 'w', encoding='utf-8')
input_file = "ProData\\2019-04-10-16-59.xls"
df = pd.read_excel(input_file)
str_list = list(df['Requirement'])
basic_tokenizer = tokenization.BasicTokenizer(do_lower_case=False)

for req in str_list:
    id = 10000
    tok_list = basic_tokenizer.tokenize(req)
    f.write('id:%d, \n' % (id))
    for tok in tok_list:
        f.write(tok+",\n")
    f.write('\n')
f.close()





