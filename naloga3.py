import cv2 as cv
import numpy as np

def kmeans(slika, k=3, iteracije=10):
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    #Preoblikovanje slike 3D (visina, sirina, kanali) v 2D
    h, w, c = slika.shape
    pixels = slika.reshape(-1, c).astype(float)     # (h*w, c)

    #Skupno stevilo pikslov
    N = slika.shape[0]

    #Polje za dodelitve
    oznake = np.zeros(N, dtype=int)

    #Izracunanje centrov
    centri = izracunaj_centre(slika, 0, k, 0)

    #Iteracija K-Means
    for it in range(iteracije): 
        #Matrika razdalj velikosti N*k
        razdalje = np.zeros((N, k), dtype=float)
        #Za vsak center
        for j in range(k):
            #Za vsak piksel
            for i in range(N):
                #Vektorska razlika med pikslom in centrom
                diff = pixels[i] - centri[j]

                #Evklidska razdalja
                razdalje[i, j] = np.sqrt(pow(diff[0], 2) + pow(diff[1], 2) + pow(diff[2], 2))
        
        #Najblizji center za vsak piksel
        move = np.argmin(razdalje, axis=1)

    pass

def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    pass

def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.'''
    pass

if __name__ == "__main__":
    pass