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

## Building
```sh
mkdir -p build
cd build
cmake ../src
make
```

## Testing
### Sequential
```sh
build/sequential/tdes_sequential tests/inFile1.txt outFile1.txt `cat keys/key1.txt`
...
build/sequential/tdes_sequential tests/inFile5.txt outFile3.txt `cat keys/key1.txt`
```

### OpenMP
```sh
build/openmp/tdes_openmp tests/inFile1.txt outFile1.txt `cat keys/key1.txt`
...
```

### CUDA
```sh
build/cuda/tdes_cuda tests/inFile1.txt outFile1.txt `cat keys/key1.txt`
...
```

## Results
For the largest file (`inFile5.txt`, 18M hex-text):

### PC 1
AMD Ryzen 5 2600X, NVidia Quadro P400, Xubuntu 18.04.1 linux 4.15.0-43-generic

#### Times
Results are sorted best to worst, 6 runs:
* Sequential: 40374.7, 40450.6, 40571.4, 40752.2, 40800.9, 40877.1
* OpenMP: 4126.18, 4137.59, 4173.01, 4178.46, 4225.43, 4331.69
* CUDA: 1091.23, 1100.22, 1101.89, 1113.01, 1119.37, 1125.29

Average of 5 best runs was used.

```
|Sequential | OpenMP    | CUDA      |
|-----------|-----------|-----------|
| 40589.96s | 4168.134s | 1105.144s |
|-----------|-----------|-----------|
| 1.0x      | 9.74x     | 36.73x    |
```
