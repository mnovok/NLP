from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

stopwords = set(STOPWORDS)
stopwords.update(["je", "i", "na", "u", "sa", "za", "po", "od", "se", "koja", "svojim", "što", "bila", "su", "kao", "zemlja", 
                  "jedno", "godine", "koje", "njihovu", "koji", "svoj", "kroz", "ima", "ostali", "koja", "kojeg", "koju", 
                  "njihove", "također", "da", "poput"])

with open('hrvatska.txt', 'r', encoding='utf-8') as file:
    text = file.read()

mask = np.array(Image.open('slika.png'))

wordcloud = WordCloud(stopwords=stopwords, background_color="white", mask=mask).generate(text)

plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()