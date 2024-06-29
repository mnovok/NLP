import nltk
from nltk.metrics.distance import edit_distance


def najbliza_rijec(trazena_rijec, rjecnik):
    min_udaljenost = float('inf')
    najbliza_rijec = None
    
    for rijec in rjecnik:
        udaljenost = edit_distance(trazena_rijec, rijec)
        if udaljenost < min_udaljenost:
            min_udaljenost = udaljenost
            najbliza_rijec = rijec
    
    return najbliza_rijec, min_udaljenost


with open("rjecnik.txt", "r", encoding="utf-8") as file:
    rijeci_iz_rjecnika = [line.strip() for line in file.readlines()]

for rijec in rijeci_iz_rjecnika:
    print(rijec)

trazena_rijec = input("Unesite neku rijec:").strip()

najbliza_rijec, udaljenost = najbliza_rijec(trazena_rijec=trazena_rijec, rjecnik=rijeci_iz_rjecnika)

if udaljenost > 0 and udaljenost <= 1:  
    odgovor = input(f"Jeste li mislili '{najbliza_rijec}' umjesto '{trazena_rijec}'? (da/ne): ").strip().lower()
    if odgovor == 'da':
        trazena_rijec = najbliza_rijec

print(f"\nNajbliza rijec u rjecniku je: {najbliza_rijec}")
