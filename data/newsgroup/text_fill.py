import time

from random import randint

start_time = time.time()

N_CYCLES = 10000
rus_ch = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
ch_num_sym = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ1234567890,.[]}{()!?'
#ch_num_sym = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890,.[]}{()!?'

num = '0123456789'
num_sym = '012345678901234567890123456789,.:;[]}{()!$%&-+<=>?@'

'abcdefghijklmnopqrstuvwxyz0123456789!$%&()+,.:;<=>?@[]{}'
'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!$%&()+,.:;<=>?@[]{}'
'абвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789!$%&()+,.:;<=>?@[]{}'
'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789!$%&()+,.:;<=>?@[]{}'


with open('newsgroup.txt', 'w') as f_tx:
    for i in range(N_CYCLES):
        # буквы + символы + цифры
        for word_ch_sym_num in range(randint(1, 20)):    
            stripe = ''
            for char in range(randint(4, 20)):
                stripe += ch_num_sym[randint(0, len(ch_num_sym) - 1)]
            f_tx.write(stripe + ' ')
        print(i, stripe)
        f_tx.write('\n')

end_time = time.time()
print('time spent = ', end_time - start_time)