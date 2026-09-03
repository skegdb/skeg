
## 13. Audit 16 (owner, 2026-09-03) su `b72ca90`: 4 P0 nuovi prima del tag - esecuzione

Verdetto 6/10: P0.1-3 e P0.5 chiusi; P0.4 chiuso per ingress e memoria trattenuta, non per il
picco egress. P0-A reply allocata prima del budget (governor osserva, non previene); P0-B VDEL
di id assenti = tombstone senza tetto, fuori da `max_vectors`; P0-C `max_disk_bytes` non copre
i blob payload; P0-D `release.yml` tratta "already exists" come successo anche con directory
cambiata e versione vecchia; `skeg-rigging-skeg 0.1.4` dipende da `skeg-vector = "0.1"` →
un 0.2 richiede nuova rigging prima di multi-tenant.
Esecuzione: R1/R2/R3 in parallelo su tre worktree da `b72ca90` (`fix/r1-egress-reservation`,
`fix/r2-user-delete-no-tombstone`, `fix/r3-disk-quota-blobs`), stesso ciclo TDD + audit a due
round; merge in sequenza R2, R3, R1 (conflitti in shard.rs risolti al merge, suite a ogni
passo). R5 (matrice versioni + validation job crates.io) parte dalla ricognizione in corso.
Prompt in scratchpad r1/r2/r3-prompt.md.
Suite su `80a2600` (bump + ponte), 65 target: seriale 1205/0/12, parallela 1205/0/12
(17:54-17:59 UTC). Harness package sulle versioni nuove in corso.
