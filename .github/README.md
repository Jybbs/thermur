<div align="center">

<a href="https://thermur.fyi/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://thermur.fyi/wordmark-dark.png"><img alt="Thermur" src="https://thermur.fyi/wordmark-light.png" width="395"></picture></a>

### *Teaching Drone Flocks to See Fire Like Starlings See Hawks*

[![Python 3.15+](https://img.shields.io/badge/python-3.15+-995500?style=for-the-badge)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-995500?style=for-the-badge)](../LICENSE)

</div>

*Thermur* flies a flock of drones the way a starling flock flies around a hawk, with fire as the hawk.

## ⌨️ Quick Start

```bash
git clone https://github.com/Jybbs/thermur.git
mise trust thermur
mise -C thermur install
cd thermur
thermur fetch sample
thermur fly
mise doctor project
```

The clone's `.mise/bin/` joins the path once mise installs, and `.mise/bin/thermur` builds `.venv` from `uv.lock` the first time it runs. `thermur fetch sample` downloads the sample, `thermur fly` flies the flock through the parametric threat and prints what the flight measures, and `mise doctor project` reports any condition the clone still lacks beside the command that supplies it.

## 🐦‍⬛ Read the Site

<p align="center">
<a href="https://thermur.fyi/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://thermur.fyi/readme/watch-dark.svg"><img alt="Watch: Thermur" src="https://thermur.fyi/readme/watch-light.svg" width="260" height="116"></picture></a>
<a href="https://thermur.fyi/data/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://thermur.fyi/readme/fetch-dark.svg"><img alt="Fetch: Data" src="https://thermur.fyi/readme/fetch-light.svg" width="260" height="116"></picture></a>
<a href="https://thermur.fyi/methods/rules"><picture><source media="(prefers-color-scheme: dark)" srcset="https://thermur.fyi/readme/fly-dark.svg"><img alt="Fly: The Interaction Rules" src="https://thermur.fyi/readme/fly-light.svg" width="260" height="116"></picture></a>
</p>

**[Data](https://thermur.fyi/data/)** names the simulations the air comes from, **[Methods](https://thermur.fyi/methods/)** sets out the rules a drone steers by, and **[Reference](https://thermur.fyi/reference/)** holds the visual identity and the kit the site's widgets are built from.

## 🙏 Credit

The air the flock flies through comes from Moisseeva and Stull's [WRF-SFIRE LES Synthetic Wildfire Plume Dataset](https://doi.org/10.20383/102.0314), published under CC BY 4.0. To cite *Thermur*, use the *"Cite this repository"* link in the sidebar, built from [`CITATION.cff`](../CITATION.cff), and [`CONTRIBUTING.md`](CONTRIBUTING.md) covers contributing.

## ©️ License

*Thermur* is released under the MIT License, whose full text is in [`LICENSE`](../LICENSE).
