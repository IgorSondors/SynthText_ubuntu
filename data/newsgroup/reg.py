# -*- coding: utf-8-sig -*-
import os
import re
import codecs

################### Делим annotation.txt на train, test, val .txt

f = codecs.open( "newsgroup_old.txt", "r", "utf8" )

fd = f.readlines()

new_fd = list(fd)

x = len(new_fd)

train = open("newsgroup.txt",'w', encoding="utf8")
for i in new_fd:
    
    newstring = re.sub(r"[^а-яА-Я0-9 ]+", "", i)
    newstring = re.sub('(  )', ' ', newstring)
    newstring = newstring.replace("\r\n", " ")
    train.write(newstring + "\n")
