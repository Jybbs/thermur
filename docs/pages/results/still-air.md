# The Flock in Still Air

*Thermur* sets out to fly a murmuration, so this page flies its flock through air with neither fire nor wind at six sizes from 10 to 3,000 drones and sets each statistic beside what was measured on starling flocks and on an outdoor flock of drones. Every statistic is computed on the flights the pipeline flies, and every norm is declared once in `thermur.metrics` and read from there.

## The Norms

Each norm names the statistic it is set beside, its figure and the spread the study states about it, the flocks it spans, and the study's DOI. Polarization is the length of the mean heading, $`0.96 \pm 0.03`$ across 24 flocks[^cavagna2010]. The speed correlation is the mean over pairs a distance $`r`$ apart of the product of their speed fluctuations about the flock's mean speed, and it first crosses zero at $`0.36\,L`$ across 24 flocks, $`L`$ the largest distance between two birds, so the length over which speeds correlate grows with the flock[^cavagna2010]. Across flocks of 10 to 3,000 starlings the group's mean speed shows no dependence on the number of birds, a Spearman rank correlation of $`-0.13`$[^cavagna2022]. A turn crosses a starling flock at 20 to 40 m/s across 12 flocks[^attanasi2014]. The flock's middle axis over its thinnest measures $`2.8 \pm 0.4`$ and its longest over its thinnest $`5.6 \pm 1.0`$ at 95% confidence across 10 flocks, and the border of a flock is denser than its inside[^ballerini2008b]. An outdoor flock of 30 drones holds its nearest neighbors 12 to 30 m apart[^vasarhelyi2018].

The table is written from `norms.toml`, and `mise render` rewrites it.

<!-- norms -->

| **Statistic** | **Norm** | **Spread** | **Flocks** | **System** | **Source** |
|---|---|---|---|---|---|
| `aspect_ratio` | 5.6 | ±1, 95% confidence interval | 10 | starlings | [10.1016/j.anbehav.2008.02.004](https://doi.org/10.1016/j.anbehav.2008.02.004) |
| `border_density` | above 1 |  | 10 | starlings | [10.1016/j.anbehav.2008.02.004](https://doi.org/10.1016/j.anbehav.2008.02.004) |
| `middle_aspect_ratio` | 2.8 | ±0.4, 95% confidence interval | 10 | starlings | [10.1016/j.anbehav.2008.02.004](https://doi.org/10.1016/j.anbehav.2008.02.004) |
| `polarization` | 0.96 | ±0.03, standard deviation | 24 | starlings | [10.1073/pnas.1005766107](https://doi.org/10.1073/pnas.1005766107) |
| `spacing_m` | 21 m | ±9, half range | 1 | drones | [10.1126/scirobotics.aat3536](https://doi.org/10.1126/scirobotics.aat3536) |
| `speed_by_size` | -0.13 |  |  | starlings | [10.1038/s41467-022-29883-4](https://doi.org/10.1038/s41467-022-29883-4) |
| `speed_correlation_ratio` | 0.36 |  | 24 | starlings | [10.1073/pnas.1005766107](https://doi.org/10.1073/pnas.1005766107) |
| `wave_speed_m_s` | 30 m/s | ±10, half range | 12 | starlings | [10.1038/nphys3035](https://doi.org/10.1038/nphys3035) |

<!-- /norms -->

## The Sweep

`Sweep` flies each size with its seeds batched together, first over a short pilot flight and then over a run whose length the pilot sets. A run spans ten of the slower of two relaxation times past its first 20 s, the heading's autocorrelation time and the time over which each drone's seven neighbors rearrange, the alignment and network relaxation times a starling flock shows[^mora2016]. Each time is the integral of its autocorrelation up to the curve's first zero, which leaves the noise past that zero out of the sum. Every size then flies again with each drone held at its cruise speed, which leaves no speed to fluctuate, so its speed correlation is the negative control a measured correlation is read against[^cavagna2010]. The table is written from `Sweep`, and `mise render` rewrites it.

<!-- sweep -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `longest_s` | `600.0` | above 0 | The longest run past the settling a size flies, in seconds, which holds the sweep's wall time and the frames a size keeps in memory bounded |
| `pilot_s` | `120.0` | above 0 | The length of the pilot flight each size's relaxation times are read from, and the shortest run past the settling a size flies, in seconds |
| `pilot_seeds` | `2` | at least 1 | The launches each size's pilot flight flies |
| `seeds` | `10` | at least 1 | The launches each size flies together under `vmap` |
| `sizes` | `[10, 30, 100, 300, 1000, 3000]` |  | The flock sizes, about evenly spaced on a logarithmic axis |
| `spans` | `10.0` | above 0 | The relaxation times a run spans past the settling |

<!-- /sweep -->

Each statistic is measured on the drones' velocities through the air, which still air leaves equal to their velocities over the ground. The speed correlation sums the products of each pair's fluctuations within each of 32 bins across the flock's extent through a weighted `KDTree.count_neighbors`, divides by the pairs in each bin and by the variance of the fluctuations, and reads the zero between the two bins about it, averaged over frames a second apart. The shape reads the axes of the flock's gyration tensor, a stand-in for the axes the source measures from the border of each flock[^ballerini2008b]. The border is that of an α-shape over the flock's `Delaunay` triangulation, keeping each tetrahedron whose circumscribed sphere is narrower than three launch spacings, 45 m, the hop under which two drones fly in one group, and the density at each drone is the inverse cube of its nearest-neighbor distance. The source names no α, so the border, and with it the density ratio, moves with that choice.

The turning wave follows the source's method[^attanasi2014]. Within the frames where the flock's mean heading turns fastest, each pair's delay is the lag maximizing the normalized cross-correlation of their radial accelerations, each drone ranks by how many drones it leads less how many lead it, and its turning time is the mean, over each drone ranked ahead, of that drone's time plus the pair's delay. The distance the turn has crossed is the cube root of the drones turned over the flock's density inside its α-shape, and its speed is the slope of that distance against the turning time over the first half of the ranks. The radio's connectivity is the share of the flock in its largest group, two drones joined wherever they fly within the radio's 50 m range.

[^cavagna2010]: Cavagna et al. 2010. "Scale-free correlations in starling flocks." *Proceedings of the National Academy of Sciences* 107(26): 11865–11870. <https://doi.org/10.1073/pnas.1005766107>

[^cavagna2022]: Cavagna et al. 2022. "Marginal speed confinement resolves the conflict between correlation and control in collective behaviour." *Nature Communications* 13(1): 2315. <https://doi.org/10.1038/s41467-022-29883-4>

[^attanasi2014]: Attanasi et al. 2014. "Information transfer and behavioural inertia in starling flocks." *Nature Physics* 10(9): 691–696. <https://doi.org/10.1038/nphys3035>

[^ballerini2008b]: Ballerini et al. 2008. "Empirical investigation of starling flocks: a benchmark study in collective animal behaviour." *Animal Behaviour* 76(1): 201–215. <https://doi.org/10.1016/j.anbehav.2008.02.004>

[^vasarhelyi2018]: Vásárhelyi et al. 2018. "Optimized flocking of autonomous drones in confined environments." *Science Robotics* 3(20): eaat3536. <https://doi.org/10.1126/scirobotics.aat3536>

[^mora2016]: Mora et al. 2016. "Local equilibrium in bird flocks." *Nature Physics* 12(12): 1153–1157. <https://doi.org/10.1038/nphys3846>
