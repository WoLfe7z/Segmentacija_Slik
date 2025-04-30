import cv2 as cv
import numpy as np

def kmeans(slika, k=3, iteracije=10):
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    #Preoblikovanje slike 3D (visina, sirina, kanali) v 2D
    M, N, C = slika.shape
    pixels = slika.reshape(-1, C).astype(float)     # (h*w, c)

    #Skupno stevilo pikslov
    st_pix = pixels.shape[0]

    #Oznaka klastra za vsak piksel
    oznake = np.zeros(st_pix, dtype=int)

    #Izracunanje centrov
    centri = izracunaj_centre(slika, 1, k, 50.0)

    #Iteracija K-Means
    for it in range(iteracije): 
        #Matrika razdalj velikosti N*k
        razdalje = np.zeros((st_pix, k), dtype=float)
        #Za vsak center
        for j in range(k):
            #Za vsak piksel
            for i in range(st_pix):
                #Vektorska razlika med pikslom in centrom
                diff = pixels[i] - centri[j]

                #Evklidska razdalja
                razdalje[i, j] = np.sqrt(pow(diff[0], 2) + pow(diff[1], 2) + pow(diff[2], 2))
        
        #Najblizji center za vsak piksel
        nove = np.argmin(razdalje, axis=1)

        #Ce ni prislo do spremembe, preskocimo, drugace posodobimo oznake klastrov
        if np.array_equal(oznake, nove):
            break
        else:
            oznake = nove

        #Posodobitev centrov kot povprecje pikslov v vsakem klastru
        for j in range(k):
            #Maska pikslov, ki so dodeljene temu centru
            mask = (oznake == j)

            #Ce ima center kakse piksle
            if np.any(mask):
                #Dodelimo nov center, ki je povprecje teh pikslov
                centri[j] = pixels[mask].mean(axis=0)
    
    #Rekonstrukcija slike
    out = centri[oznake]
    slika_out = out.reshape(M, N, 3).astype(np.uint8)
    return slika_out

def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    pass

def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.'''
    #Prebere dimenzije slike in pripracimo vektor
    M, N, C = slika.shape
    pixels = slika.reshape(-1, C).astype(float)  

    #Ce je dimenzija_centra = 3, upostevamo barvo, ce ne potem upostevamo lokacije centrov in barve
    if(dimenzija_centra == 3):
        #Samo barve (M*N, C)
        oznake = pixels
    else:
        #Izracunamo x in y za vsak piksel
        #Oznaka za vsako vrsto od 0 do M*N-1
        idx = np.arange(pixels.shape[0])

        #Stolpec, 
        x = idx % N

        #Vrsica, 
        y = idx // N

        coords = np.stack([x, y], axis=1).astype(float)
        oznake = np.hstack(pixels, coords)

    #Stevilo centrov
    k = dimenzija_centra
    #Seznam za shranjevanje centrov
    centri = []

    #Nakljucno
    if izbira == 0:
        #Naklucno stevilo, ki bo sluzilo kot prvi center
        prvi = np.random.randint(len(oznake))
        centri.append(oznake[prvi])

        #Nato izbiramo dokler nimamo k centrov
        while len(centri) < k:
            idx = np.random.randint(len(oznake))
            #Random novi center
            nov = oznake[idx]
            #Racunanje razdalje do vseh izbranih centrov
            razdalje = []

            for c in centri:
                diff = nov - c
                #Evklidksa razdalja
                razdalje_val = np.sqrt(pow(diff[0], 2) + pow(diff[1], 2) + pow(diff[2], 2))
                razdalje.append(razdalje_val)
                
                if min(razdalje) >= T:
                    centri.append(nov)

        return np.array(centri)
    #Rocno
    else:
        izbrani_koord = []

        #Ob kliku klici funkcijo
        def mouse_cb(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN and len(izbrani_koord) < k:
                #Shrani kliknjeno mesto
                izbrani_koord.append((x,y))

                #Ko imamo k klikov, zapri okno
                if len(izbrani_koord) == k:
                    cv.destroyAllWindows()
        
        #Prikazemo okno in cakamo na klike
        cv.namedWindow('Klikni centre')
        cv.setMouseCallback('Klikni centre', mouse_cb)

        while len(izbrani_koord) < k:
            cv.imshow('Klikni centre', slika.astype(np.uint8))
            cv.waitKey(50)
        
        #Vsak klik se pretvori v vektor oznak
        for x, y in izbrani_koord:
            indeks = y * N + x
            centri.append(oznake[indeks])

        return np.array(centri)

if __name__ == "__main__":
    slika = cv.imread('.utils/zelenjava.jpg')

    rezultat = kmeans(slika, k=3, iteracije=10)
    cv.imshow('Segmentirana slika', rezultat)
    cv.waitKey(0)
    cv.destroyAllWindows()
    pass