import pandas as pd
import numpy as np
import re

with open('text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

## Broj rijeci koje pocinju velikim slovom
capitalized = r'\b[A-ZČĆĐŽŠ][a-zčćđžš]*[0-9]*\b'
words = re.findall(capitalized, text)
print(words)

## Korisnik trazi rijec
print("Unesite zeljenu rijec: ")
search = input()
regex = r'\b' + re.escape(search) + r'\b'
word = re.findall(regex, text, re.IGNORECASE)
occurrences = len(word)
print(f"Rijec '{search}' se pojavila {occurrences} puta u tekstu.")

## MM/DD/GGGG se pretvara u DD/MM/GGGG
old_date = r'\b(\d{1,2})\/(\d{1,2})\/(\d{4})\b'

def convert_date(date):
    month = date.group(1)
    day = date.group(2)
    year = date.group(3)
    return f"{day}/{month}/{year}"

converted_text = re.sub(old_date, convert_date, text)

text_file = open("new_dates.txt", "w", encoding="utf-8")
n = text_file.write(converted_text)
text_file.close()

print("Datumi izmijenjeni u novoj datoteci.")