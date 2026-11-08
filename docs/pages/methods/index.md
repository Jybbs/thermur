# Methods

The pages in this section set out how *Thermur* flies its flock, meaning what each drone senses, the rules it steers by, how a flight is integrated, and the measurements that decide which rule fills each part of the controller. Each drone reads only its own state, its seven nearest neighbors, and the air at its position, so a drone's action depends on nothing it cannot sense, and the suite holds every rule to that.

- [The Interaction Rules](rules.md) describes each rule a drone sums, the physical system each one is borrowed from, and the flights that chose among them

`thermur fly` flies the reference flight through the parametric threat and prints its statistics, and each run it records lands under `data/runs/`, where `mise browse` opens Hamilton's experiment viewer over them.
