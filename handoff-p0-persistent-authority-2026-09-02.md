
### 16.8 Audit 25 (B1), merge B5 - 4 settembre

Audit 25 (opus, secure-code-review): merge condizionato. Invariante B1 vero (misura
dall'indice → prenota → fetch, contatore al worker). F1 Medium CWE-703: SET concorrente che
ingrandisce il valore → GET legittimo in `-ERR` senza retry (14/4000 misurati); F2 Medium
CWE-400: nessun cap di arità su MGET e strutture del preflight allocate prima del charge
(~200 B/chiave vs 9 sul filo); F3/F4 test deboli; B7 deterministico per costruzione su
`ok+refused == BURST`. Fix mandati all'implementatore (retry singolo con ri-misura,
`MAX_MGET_KEYS` + charge del preflight, test). Integrate dopo B0: server+core 571/0. Merge
B5 in integrate; gate workflow rieseguiti.
