import cv2 as cv
import numpy as np

def kmeans(slika, k, iteracije, T, dim):
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    #Preoblikovanje slike 3D (visina, sirina, kanali) v 2D
    M, N, C = slika.shape
    pixels = slika.reshape(-1, C).astype(float)     #(h*w, c), -1 samo zracuna dimenzija M*N

    #Skupno stevilo pikslov
    st_pix = pixels.shape[0]

    #Oznaka klastra za vsak piksel
    oznake = np.zeros(st_pix, dtype=int)

    izbira = 0  #Nakljucno
    #Izracunanje centrov
    centri = izracunaj_centre(slika, izbira, dim, T, k)

    if(dim > 3):
        #Izracunamo x in y za vsak piksel
        idx = np.arange(pixels.shape[0])    #Ustvari zaporedje pikslov
        x = idx % N     #X koordinata: stolpci
        y = idx // N    #Y koordinata: vrstice (// vrne brez ostanka, ker se uporablja pri racunanju stolpcev)
        coords = np.stack([x, y], axis=1).astype(float) #Zdruzi x in y v en array
        oznake = np.hstack((pixels, coords))            #Zdruzi barvo in koordinate

    #Iteracija K-Means
    for it in range(iteracije): 
        #Evklidska razdalja med vsakim pikslom in vsakim centrom
        #axis = 2 racuna po vseh znacilnicah (dim=3 samo po barvi, dim=5 po barvi in poziciji)
        #centri[k, dim] in piksli[N, dim] dodamo novo dimenzijo(s None), da potem np ustvari 3d matriko(N, k, dim)
        razdalje = np.linalg.norm(pixels[:, None, :] - centri[None, :, :], axis=2)
        
        #Vrne indeks za najblizji center za vsak piksel (Racuna po vrsticah)
        nove = np.argmin(razdalje, axis=1)

        #Ce ni prislo do spremembe, preskocimo, drugace posodobimo oznake klastrov
        if np.array_equal(oznake, nove):
            break

        #Shranimo nove spremembe v oznake
        oznake = nove

        #Posodobitev centrov kot povprecje pikslov v vsakem klastru
        for j in range(k):
            #Maska pikslov, ki so dodeljene temu centru
            mask = (oznake == j)

            #Ce ima center kakse piksle
            if np.any(mask):
                #Dodelimo nov center, ki je povprecje teh pikslov
                centri[j] = pixels[mask].mean(axis=0)
    
    #Za vsak indeks v oznakah vzame ustrezno barvo iz centrov
    out = centri[oznake]
    #[:, :3] pretvori nazaj v RGB kanale, reshape(M,N,3) vrne nazaj 3d sliko, np.uint8 spremeni nazaj v cele stevilke od 0 do 255
    slika_out = out[:, :3].reshape(M, N, 3).astype(np.uint8)
    return slika_out

#Funkcija za izracun razdalje med dvema tockama
def izracunaj_razdaljo(x, oznake):
    #Izracuna in vrne evklidsko razdaljo med tocko x in vsako tocko v oznakah, axis=1 racuna po vrsticah
    return np.linalg.norm(oznake - x, axis=1)

#Funkcija za izracun gaussovega jedra
def gaussovo_jedro(d, h):
    return np.exp(-(pow(d, 2) / (2 * pow(h, 2))))

#Funkcija za preverjanje konvergence
def preveri_konvergenco(x_new, x, eps):
    #Preveri, ce je premik tocke manj kot eps
    return np.linalg.norm(x_new - x) < eps

def meanshift(slika, velikost_okna, dimenzija, min_cd):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    #Parametri
    max_iter = 5    #Max iteracij za vsak piksel
    eps = 1e-3     #Prag za konvergenco (1x10^(-3)), ce je prevelik, konvergenca ne dobi pravih podatkov

    #Prebere dimenzije slike in pripravimo vektor
    M, N, C = slika.shape
    pixels = slika.reshape(-1, C).astype(float)     # (h*w, c), -1 samo zracuna dimenzija M*N

    #Ce je dimenzija = 3, upostevamo samo barvo, ce ne potem upostevamo lokacije centrov in barve
    if(dimenzija == 3):
        #Samo barve (M*N, C)
        oznake = pixels
    else:
        #Izracunamo x in y za vsak piksel
        idx = np.arange(pixels.shape[0])    #Ustvari zaporedje pikslov
        x = idx % N     #X koordinata: stolpci
        y = idx // N    #Y koordinata: vrstice (// vrne brez ostanka, ker se uporablja pri racunanju stolpcev)
        coords = np.stack([x, y], axis=1).astype(float) #Zdruzi x in y v en array
        oznake = np.hstack((pixels, coords))            #Zdruzi barvo in koordinate

    #Matrika za shranjevanje premaknjenih tock
    shifted = np.zeros_like(oznake)

    #Za vsak piksel v vektorju oznake
    for i in range(len(oznake)):
        x = oznake[i]   #Trenutna tock
        iteracija = 0   #St iteracij
        konvergenca = False
        while not konvergenca and iteracija < max_iter:
            #Izracun razdalj do vseh tock
            razdalje = izracunaj_razdaljo(x, oznake)

            #Utezi z jedrom in velikost okna
            utezi = gaussovo_jedro(razdalje, velikost_okna)

            #utezi[:, None], ker so oznake v obliki (M), axis=0 steje po stolpcih
            x_new = np.sum(utezi[:, None] * oznake, axis=0) / np.sum(utezi)

            #Preveri konvergenco med tockami
            konvergenca = preveri_konvergenco(x_new, x, eps)
            x = x_new
            iteracija += 1
        shifted[i] = x

    #Zdruzi konvergirane tocke v centre
    centri = []
    for x in shifted:
        if not centri:
            centri.append(x)    #Prvi center vedno dodamo
        else:
            dc = izracunaj_razdaljo(x, np.array(centri))    #Razdalja do vseh ze obstojecih centrov
            #Ce je tocka dovolj dalec od sotalih centrov jo dodamo kot nov center
            if min(dc) >= min_cd:
                centri.append(x)
    
    centri = np.array(centri)

    #Dodeli vsako tocko shifted tocku najblizjemu centru
    labels = np.zeros(len(shifted), dtype=int)
    for i in range(len(shifted)):
        dc = izracunaj_razdaljo(shifted[i], centri)     #Razdalje med tocko in centri
        labels[i] = np.argmin(dc)   #Vrne indeks namanjse vrednosti (najkrajse razdalje)

    #Za vsak indeks v labels vzame ustrezno barvo iz centrov
    seg_pixels = centri[labels]
    #[:, :3] pretvori nazaj v RGB kanale, reshape(M,N,3) vrne nazaj 3d sliko, np.uint8 spremeni nazaj v cele stevilke od 0 do 255
    out = seg_pixels[:, :3].reshape(M, N, 3).astype(np.uint8)
    return out

def izracunaj_centre(slika, izbira, dimenzija_centra, T, k):
    '''Izračuna centre za metodo kmeans.'''
    #Prebere dimenzije slike in pripravimo vektor
    M, N, C = slika.shape
    pixels = slika.reshape(-1, C).astype(float)     #(h*w, c), -1 samo zracuna dimenzija M*N  

    #Ce je dimenzija_centra = 3, upostevamo samo barvo, ce ne potem upostevamo lokacije centrov in barve
    if(dimenzija_centra == 3):
        #Samo barve (M*N, C)
        oznake = pixels.copy()
    else:
        #Izracunamo x in y za vsak piksel
        idx = np.arange(pixels.shape[0])    #Ustvari zaporedje pikslov
        x = idx % N     #X koordinata: stolpci
        y = idx // N    #Y koordinata: vrstice (// vrne brez ostanka, ker se uporablja pri racunanju stolpcev)
        coords = np.stack([x, y], axis=1).astype(float) #Zdruzi x in y v en array
        oznake = np.hstack((pixels, coords))            #Zdruzi barvo in koordinate

    #Seznam za shranjevanje centrov
    centri = []

    #Nakljucno
    if izbira == 0:
        #Naklucno stevilo, ki bo sluzilo kot prvi center, ter ga doda v centre
        prvi = np.random.randint(len(oznake))
        centri.append(oznake[prvi])

        #Nato izbiramo dokler nimamo k centrov
        while len(centri) < k:
            nov = oznake[np.random.randint(len(oznake))]

            #Evklidska razdalja random tocke do centrov
            razdalje = np.linalg.norm(np.array(centri) - nov, axis=1)

            #Dodamo nov center, ce je dovolj oddaljen od vseh
            if min(razdalje) >= T:
                centri.append(nov)

        return np.array(centri)
    #Rocno
    else:
        izbrani_koord = []

        #Ob kliku klici funkcijo
        def mouse_cb(event, x, y):
            if event == cv.EVENT_LBUTTONDOWN and len(izbrani_koord) < k:
                #Shrani kliknjeno mesto
                izbrani_koord.append((x,y))

                #Ko imamo k klikov, zapri okno
                if len(izbrani_koord) == k:
                    cv.destroyAllWindows()
        
        #Prikazemo okno in cakamo na klike
        cv.namedWindow('Klikni centre') #namedWindow naredi okno, ki omogoca interakcijo s misko
        cv.setMouseCallback('Klikni centre', mouse_cb)

        #Dokler ni dovolj klikov prikazuje sliko
        while len(izbrani_koord) < k:
            cv.imshow('Klikni centre', slika.astype(np.uint8))
            cv.waitKey(50)  #Pocaka 50 ms pred naslednjim klikom
        
        #Vsak klik se pretvori v vektor oznak
        for x, y in izbrani_koord:
            indeks = y * N + x  #Pretvorba iz 2D v 1D (v vrstico)
            #Dodamo kliknjen piksel med centre
            centri.append(oznake[indeks])

        return np.array(centri)

if __name__ == "__main__":
    slika = cv.imread('.utils/zelenjava.jpg')
    slika = cv.resize(slika, (200, 200))

    #kmeans testiranje
    #kmeans_alg = kmeans(slika, 10, 10, 30.0, 3)
    #cv.imshow('kmeans_dim = 3', kmeans_alg)
    #kmeans_alg1 = kmeans(slika, 10, 10, 30.0, 5)
    #cv.imshow('kmeans_dim = 5', kmeans_alg1)

    #mean-shift testiranje
    #meanshift_alg10 = meanshift(slika, 50.0, 5, 10.0)
    #cv.imshow('mean-shift_10', meanshift_alg10)
    #meanshift_alg20 = meanshift(slika, 50.0, 5, 50.0)
    #cv.imshow('mean-shift_50', meanshift_alg20)

    cv.waitKey(0)
    cv.destroyAllWindows()
    pass