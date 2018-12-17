# 3des-ecb

## Task
Algorytm szyfrujący 3DES w trybie ECB (Triple Data Encryption Standard) w trybie szyfrowania ECB (Electronic Code Book).
Symetryczny algorytm z kluczem.
Szyfruje i deszyfruje dane w 64-bitowych blokach wykorzystując 56-bitowe klucze.
Działanie polega na trzykrotnym przetworzeniu danych klasycznym algorytmem DES (szyfruje pierwszym kluczem, deszyfruje drugim, szyfruje trzecim).
Należy zapoznać się z algorytmem, zrealizować w wersji sekwencyjnej.
Wybrać tryb szyfrowania najlepszy dla zrównoleglenia.
Następnie zaproponować sposób zrównoleglenia i zrealizować w dwóch wersjach.

## Comparing outputs with OpenSSL
```
openssl des-ede3 -K <key> -in <infile> -out <outfile>
```

## Generating keys
```
openssl enc -des-ede3 -pass pass:<some_password> -p -in testInFile.txt -out testOutFile.txt
```
And save the displayed key.
Example key saved in `keys/key1.txt`.
